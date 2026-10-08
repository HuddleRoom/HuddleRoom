"""End-to-end tests for the authorized-execution loop in tick()
(OrchestrationService._advance_authorized_execution). Drives tick() directly
on the seeded db_session (never a second AsyncSessionLocal connection) and
stubs the LLM decision adapter -- no real provider calls. See
huddleroom/services/orchestration_service.py::_advance_authorized_execution.
"""
import json
import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from fastapi import HTTPException

from huddleroom.config import settings
from huddleroom.models.agent import Agent
from huddleroom.models.artifact import Artifact
from huddleroom.models.base import _utcnow
from huddleroom.models.event_log import EventLog
from huddleroom.models.meeting import Meeting
from huddleroom.models.orchestration import (
    OrchestrationAction,
    OrchestrationDecision,
    OrchestrationEvidence,
    OrchestrationGate,
    OrchestrationGoal,
    OrchestrationRun,
)
from huddleroom.models.orchestration_process import OrchestrationWarning
from huddleroom.models.graph import Graph, GraphRun
from huddleroom.models.project import Project
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.schemas.orchestration import OrchestrationGoalCreate, OrchestrationPlanItem
from huddleroom.services.event_bus import emit_event, emit_event_once
from huddleroom.services.artifact_service import ArtifactService
from huddleroom.services.orchestration_decision_dispatcher import OrchestrationDecisionDispatcher
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.orchestration_effectiveness_review import EffectivenessReviewProcess
from huddleroom.services.orchestration_memory_service import OrchestrationMemoryService
from huddleroom.services.task_service import TaskService
from huddleroom.services.orchestration_work_report import parse_work_report

from tests.conftest import complete_baseline_processes, heal_baseline_drift_for_test
from tests.orchestration_wake_helpers import NOOP_WAKE_WHEN

pytestmark = pytest.mark.asyncio


def _agent(name_prefix: str, capabilities: list[str]) -> Agent:
    return Agent(
        name=f"{name_prefix}-{uuid.uuid4()}",
        role="agent",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=capabilities,
        config={},
        is_active=True,
    )


async def _authorized_run(db_session, test_project, success_criteria=None):
    """Create a goal, drive its baseline to terminal with the conftest Safe*
    analyzers (no real LLM -- see complete_baseline_processes), tick it to
    phase=ready, then explicitly Start it (phase=authorized). Any active
    agent must exist BEFORE this call: the real manager_selection process
    records the roster it considered, and a *later* roster change (agent
    created after baseline is terminal) would trigger spec-8.1's "roster
    gained candidates" auto-rerun on every subsequent tick -- which calls
    the real (unstubbed) manager analyzer and stalls baseline_ready."""
    service = OrchestrationService()
    goal, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Ship the authorized-execution runtime loop",
            success_criteria=success_criteria or [
                {"key": "done", "description": "The objective is delivered."}
            ],
            constraints={},
            budget={},
        ),
        created_by_user_id=None,
    )
    run.baseline_authorized = True
    await complete_baseline_processes(db_session, goal, run)
    await service.tick(db_session, run.id)
    await db_session.refresh(run)
    assert run.phase == "ready"
    goal, run = await service.start_run(
        db_session, test_project.id, goal.id, actor="human:test"
    )
    assert run.phase == "authorized"
    return service, goal, run


def _planner_agent_id(ctx: dict) -> str:
    fits = ctx["roster"]["work_functions"]["planning"]
    assert fits, "expected at least one planning-capable agent in the decision roster"
    return fits[0]["agent_id"]


async def _run_actions(db_session, run_id):
    return (
        await db_session.scalars(
            select(OrchestrationAction).where(OrchestrationAction.run_id == run_id)
        )
    ).all()


async def _seed_accepted_plan(db_session, test_project, service, run, planner, plan_items=None):
    """Drive a real request_plan -> accept_plan cycle (bypassing the LLM decision
    loop) so run.plan_state["status"] == "accepted" with the given plan items
    (defaults to one expandable item)."""
    request_action = await service.execute_request_plan_action(
        db_session,
        run_id=run.id,
        request={
            "action_type": "request_plan",
            "agent_id": str(planner.id),
            "work_function": "planning",
            "scope": "Create a plan with one independently verifiable item.",
        },
        idempotency_key=f"run:{run.id}:kind:test-request_plan",
    )
    artifact = Artifact(
        project_id=test_project.id,
        name="e2e-plan",
        artifact_type="plan",
        status="draft",
        linked_task_id=request_action.target_id,
        created_by_agent=planner.id,
        metadata_={
            "kind": "implementation_plan",
            "plan_items": plan_items
            or [
                {
                    "id": "item-1",
                    "work_function": "planning",
                    "scope": "Implement the accepted plan item.",
                    "deliverable": "A code change plus test output.",
                    "agent_id": str(planner.id),
                }
            ],
        },
    )
    db_session.add(artifact)
    await db_session.flush()
    await service.execute_accept_plan_action(
        db_session,
        run_id=run.id,
        request={"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
        idempotency_key=f"run:{run.id}:kind:test-accept_plan",
    )
    await db_session.refresh(run)
    assert run.plan_state["status"] == "accepted"


async def test_plan_item_criterion_keys_are_normalized():
    item = OrchestrationPlanItem.model_validate(
        {
            "id": "item-1",
            "work_function": "planning",
            "scope": "s",
            "deliverable": "d",
            "success_criterion_keys": [" done ", "done", ""],
        }
    )

    assert item.success_criterion_keys == ["done"]


@pytest.mark.parametrize(
    ("criteria", "item_keys", "reason"),
    [
        (
            [{"key": "done", "description": "done"}],
            ["unknown"],
            "unknown success criterion key",
        ),
        (
            [
                {"key": "criterion-one", "description": "one"},
                {"key": "criterion-two", "description": "two"},
            ],
            ["criterion-one"],
            "does not cover every declared success criterion",
        ),
    ],
)
async def test_accept_plan_rejects_unknown_or_uncovered_criterion_links(
    db_session, test_project, criteria, item_keys, reason
):
    planner = _agent("criterion-plan", ["planning"])
    db_session.add(planner)
    await db_session.flush()
    service, goal, run = await _authorized_run(
        db_session, test_project, success_criteria=criteria
    )
    request_action = await service.execute_request_plan_action(
        db_session,
        run.id,
        {
            "action_type": "request_plan",
            "agent_id": str(planner.id),
            "work_function": "planning",
            "scope": "Create a criterion-linked plan.",
        },
        f"run:{run.id}:kind:criterion-request",
    )
    artifact = Artifact(
        project_id=test_project.id,
        name="criterion-plan",
        artifact_type="plan",
        status="draft",
        linked_task_id=request_action.target_id,
        created_by_agent=planner.id,
        metadata_={"plan_items": [{
            "id": "item-1", "work_function": "planning", "scope": "s", "deliverable": "d",
            "success_criterion_keys": item_keys,
        }]},
    )
    db_session.add(artifact)
    await db_session.flush()

    with pytest.raises(HTTPException, match=reason):
        await service.execute_accept_plan_action(
            db_session, run.id,
            {"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
            f"run:{run.id}:kind:criterion-accept",
        )

    await db_session.refresh(run)
    assert run.plan_state["status"] == "revision_required"

@pytest.mark.parametrize(
    ("plan_items", "reason"),
    [
        ([{"id": "broken"}], "Invalid plan item"),
        (
            [
                {"id": "same", "work_function": "planning", "scope": "s", "deliverable": "d"},
                {"id": "same", "work_function": "planning", "scope": "s", "deliverable": "d"},
            ],
            "Duplicate plan item id",
        ),
        ([{"id": "one", "work_function": "planning", "scope": "s", "deliverable": "d", "depends_on": ["missing"]}], "unknown plan item"),
        ([{"id": "one", "work_function": "planning", "scope": "s", "deliverable": "d", "depends_on": ["one"]}], "cannot depend on itself"),
        (
            [
                {"id": "one", "work_function": "planning", "scope": "s", "deliverable": "d", "depends_on": ["two"]},
                {"id": "two", "work_function": "planning", "scope": "s", "deliverable": "d", "depends_on": ["one"]},
            ],
            "cycle",
        ),
    ],
)
async def test_malformed_plan_requires_revision_then_accepts_corrected_plan(
    db_session, test_project, plan_items, reason
):
    planner = _agent("malformed-plan", ["planning"])
    db_session.add(planner)
    await db_session.flush()
    service, _, run = await _authorized_run(db_session, test_project)
    request_action = await service.execute_request_plan_action(
        db_session,
        run_id=run.id,
        request={
            "action_type": "request_plan",
            "agent_id": str(planner.id),
            "work_function": "planning",
            "scope": "Create a valid execution plan.",
        },
        idempotency_key=f"run:{run.id}:kind:malformed-plan-request",
    )
    malformed = Artifact(
        project_id=test_project.id,
        name="malformed-plan",
        artifact_type="plan",
        status="draft",
        linked_task_id=request_action.target_id,
        created_by_agent=planner.id,
        metadata_={"plan_items": plan_items},
    )
    db_session.add(malformed)
    await db_session.flush()

    with pytest.raises(HTTPException, match=reason):
        await service.execute_accept_plan_action(
            db_session,
            run_id=run.id,
            request={"action_type": "accept_plan", "plan_artifact_id": str(malformed.id)},
            idempotency_key=f"run:{run.id}:kind:malformed-plan-accept",
        )

    await db_session.refresh(run)
    failed_accept = next(action for action in await _run_actions(db_session, run.id) if action.action_type == "accept_plan")
    assert failed_accept.status == "failed"
    assert run.plan_state["status"] == "revision_required"
    assert reason in run.plan_state["revision_reason"]

    await service.execute_request_plan_revision_action(
        db_session,
        run_id=run.id,
        request={
            "action_type": "request_plan_revision",
            "plan_task_id": str(request_action.target_id),
            "revision_request": "Repair the structural plan errors.",
        },
        idempotency_key=f"run:{run.id}:kind:malformed-plan-revision",
    )
    corrected = Artifact(
        project_id=test_project.id,
        name="corrected-plan",
        artifact_type="plan",
        status="draft",
        linked_task_id=request_action.target_id,
        created_by_agent=planner.id,
        metadata_={
            "plan_items": [{"id": "fixed", "work_function": "planning", "scope": "s", "deliverable": "d"}]
        },
    )
    db_session.add(corrected)
    await db_session.flush()
    accepted = await service.execute_accept_plan_action(
        db_session,
        run_id=run.id,
        request={"action_type": "accept_plan", "plan_artifact_id": str(corrected.id)},
        idempotency_key=f"run:{run.id}:kind:corrected-plan-accept",
    )
    await db_session.refresh(run)
    assert accepted.status == "completed"
    assert run.plan_state["status"] == "accepted"


async def test_plan_validation_handles_deep_acyclic_dag():
    depth = 1_200
    plan_items = [
        {
            "id": f"item-{index}",
            "work_function": "planning",
            "scope": "s",
            "deliverable": "d",
            "depends_on": [f"item-{index + 1}"] if index < depth - 1 else [],
        }
        for index in range(depth)
    ]
    artifact = Artifact(
        name="deep-acyclic-plan",
        artifact_type="plan",
        metadata_={"plan_items": plan_items},
    )

    items = OrchestrationService()._plan_items_from_artifact(artifact)

    assert len(items) == depth
    assert items[-1].id == f"item-{depth - 1}"


async def test_runtime_commits_malformed_plan_failure_then_accepts_revision(
    db_session, concurrent_sessions, test_project, stub_decision, monkeypatch
):
    # Step-by-step plan failure/revision contract; pin single-action ticks.
    monkeypatch.setattr(settings, "orchestration_max_actions_per_tick", 1)
    planner = _agent("runtime-malformed-plan", ["planning"])
    db_session.add(planner)
    await db_session.flush()
    service, _, run = await _authorized_run(db_session, test_project)
    request_action = await service.execute_request_plan_action(
        db_session,
        run_id=run.id,
        request={
            "action_type": "request_plan",
            "agent_id": str(planner.id),
            "work_function": "planning",
            "scope": "Create a valid execution plan.",
        },
        idempotency_key=f"run:{run.id}:kind:runtime-malformed-request",
    )
    malformed = Artifact(
        project_id=test_project.id,
        name="runtime-malformed-plan",
        artifact_type="plan",
        status="draft",
        linked_task_id=request_action.target_id,
        created_by_agent=planner.id,
        metadata_={"plan_items": [{"id": "broken"}]},
    )
    db_session.add(malformed)
    await db_session.commit()

    execution_db, reload_db = concurrent_sessions
    stub_decision(lambda _ctx: {"action_type": "accept_plan", "plan_artifact_id": str(malformed.id)})
    result = await service.tick(execution_db, run.id)
    assert result["authorized_execution"]["step"] == "plan_revision_required"

    failed_accept = await reload_db.scalar(
        select(OrchestrationAction).where(
            OrchestrationAction.run_id == run.id,
            OrchestrationAction.action_type == "accept_plan",
        )
    )
    persisted_run = await reload_db.get(OrchestrationRun, run.id)
    persisted_gate = await reload_db.scalar(
        select(OrchestrationGate).where(
            OrchestrationGate.run_id == run.id,
            OrchestrationGate.success_criterion_key == "plan",
        )
    )
    assert failed_accept.status == "failed"
    assert persisted_run.plan_state["status"] == "revision_required"
    assert persisted_gate.status == "open"
    await reload_db.rollback()

    stub_decision(
        lambda _ctx: {
            "action_type": "request_plan_revision",
            "plan_task_id": str(request_action.target_id),
            "revision_request": "Repair the malformed plan.",
        }
    )
    revision_result = await service.tick(execution_db, run.id)
    assert revision_result["authorized_execution"]["step"] == "plan_decision"
    corrected = Artifact(
        project_id=test_project.id,
        name="runtime-corrected-plan",
        artifact_type="plan",
        status="draft",
        linked_task_id=request_action.target_id,
        created_by_agent=planner.id,
        metadata_={
            "plan_items": [{"id": "fixed", "work_function": "planning", "scope": "s", "deliverable": "d"}]
        },
    )
    execution_db.add(corrected)
    await execution_db.commit()

    stub_decision(lambda _ctx: {"action_type": "accept_plan", "plan_artifact_id": str(corrected.id)})
    accepted_result = await service.tick(execution_db, run.id)
    assert accepted_result["authorized_execution"]["step"] == "plan_decision"
    await reload_db.rollback()
    accepted_run = await reload_db.get(OrchestrationRun, run.id)
    assert accepted_run.plan_state["status"] == "accepted"


async def test_runtime_reraises_invalid_revision_request_after_malformed_plan(
    db_session, test_project, stub_decision
):
    planner = _agent("invalid-runtime-revision", ["planning"])
    db_session.add(planner)
    await db_session.flush()
    service, _, run = await _authorized_run(db_session, test_project)
    request_action = await service.execute_request_plan_action(
        db_session,
        run_id=run.id,
        request={
            "action_type": "request_plan",
            "agent_id": str(planner.id),
            "work_function": "planning",
            "scope": "Create a valid execution plan.",
        },
        idempotency_key=f"run:{run.id}:kind:invalid-revision-request",
    )
    malformed = Artifact(
        project_id=test_project.id,
        name="invalid-runtime-revision-plan",
        artifact_type="plan",
        status="draft",
        linked_task_id=request_action.target_id,
        created_by_agent=planner.id,
        metadata_={"plan_items": [{"id": "broken"}]},
    )
    db_session.add(malformed)
    await db_session.flush()
    with pytest.raises(HTTPException, match="Invalid plan item"):
        await service.execute_accept_plan_action(
            db_session,
            run.id,
            {"action_type": "accept_plan", "plan_artifact_id": str(malformed.id)},
            f"run:{run.id}:kind:invalid-revision-malformed-accept",
        )

    stub_decision(
        lambda _ctx: {
            "action_type": "request_plan_revision",
            "plan_task_id": str(uuid.uuid4()),
            "revision_request": "This task id is invalid.",
        }
    )
    with pytest.raises(HTTPException, match="Planning task not found"):
        await service.tick(db_session, run.id)


async def _tasks_for_run(db_session, run_id):
    result = await db_session.execute(
        select(Task).where(Task.metadata_["orchestration"]["run_id"].as_string() == str(run_id))
    )
    return list(result.scalars().all())


async def _complete_task_session(
    db_session, project_id, task, agent_id, output, *, emit_task_event=True, event_type="session.completed",
    graph_run_id=None,
):
    """Complete an owned task through the same durable event boundary tick reads."""
    task.status = "done"
    session = Session(
        task_id=task.id,
        agent_id=agent_id,
        project_id=project_id,
        adapter_type="api",
        status="completed",
        output=output,
        graph_run_id=graph_run_id,
        metadata_={"orchestration": dict(((task.metadata_ or {}).get("orchestration") or {}))},
        origin="auto",
    )
    db_session.add(session)
    await db_session.flush()
    if emit_task_event:
        await emit_event_once(
            db_session,
            project_id,
            "task.status_changed",
            {"task_id": str(task.id), "status": "done", "previous_status": "in_progress"},
            dedup_key=f"runtime-e2e-task-done:{task.id}",
        )
    payload = {"session_id": str(session.id), "task_id": str(task.id)}
    if event_type == "review.approved":
        payload["review_outcome"] = {"verdict": "accepted"}
    await emit_event_once(
        db_session,
        project_id,
        event_type,
        payload,
        dedup_key=f"runtime-e2e-{event_type}:{session.id}",
    )
    return session


async def _completed_parent_task(db_session, test_project):
    agent = _agent("follow-up", ["implementation", "planning"])
    db_session.add(agent)
    await db_session.flush()
    service, _, run = await _authorized_run(db_session, test_project)
    action = await service.execute_create_delegation_task_action(
        db_session,
        run.id,
        {
            "action_type": "create_delegation_task",
            "agent_id": str(agent.id),
            "work_function": "implementation",
            "scope": "Deliver the parent work.",
            "deliverable": "The delivered work.",
        },
        f"run:{run.id}:kind:test-parent",
    )
    parent = await db_session.get(Task, action.target_id)
    parent.status = "done"
    report_session = Session(
        task_id=parent.id,
        agent_id=agent.id,
        project_id=test_project.id,
        adapter_type="api",
        status="completed",
        output="{\"summary\": \"Delivered parent work.\"}",
    )
    db_session.add(report_session)
    await db_session.flush()
    await service.reserve_action(
        db_session,
        run_id=run.id,
        idempotency_key=f"run:{run.id}:kind:report_consumed:task:{parent.id}",
        action_type="report_consumed",
        request={"task_id": str(parent.id), "session_id": str(report_session.id)},
    )
    return service, run, parent, agent, report_session


async def test_follow_up_creates_one_child_for_its_parent_agent(db_session, test_project):
    service, run, parent, agent, report_session = await _completed_parent_task(db_session, test_project)
    request = {
        "action_type": "create_delegation_task",
        "agent_id": str(agent.id),
        "work_function": "follow_up",
        "scope": " Answer a focused question about the delivered work. ",
        "deliverable": "A short answer.",
        "parent_task_id": str(parent.id),
        "source_session_id": str(report_session.id),
    }

    first = await service.execute_create_delegation_task_action(
        db_session, run.id, request, "first-regenerated-decision"
    )
    second = await service.execute_create_delegation_task_action(
        db_session, run.id, request, "second-regenerated-decision"
    )

    tasks = await _tasks_for_run(db_session, run.id)
    followups = [task for task in tasks if service._task_work_function(task) == "follow_up"]
    assert first.id == second.id
    assert len(followups) == 1
    assert followups[0].parent_id == parent.id
    assert followups[0].assigned_to == agent.id
    assert first.request["source_session_id"] == str(report_session.id)


async def test_follow_up_questions_have_distinct_replay_identity(db_session, test_project):
    """A new question about the same consumed report is new work, not a replay."""
    service, run, parent, agent, report_session = await _completed_parent_task(db_session, test_project)
    base = {
        "action_type": "create_delegation_task",
        "agent_id": str(agent.id),
        "work_function": "follow_up",
        "deliverable": "A short answer.",
        "parent_task_id": str(parent.id),
        "source_session_id": str(report_session.id),
    }

    first = await service.execute_create_delegation_task_action(
        db_session,
        run.id,
        {**base, "scope": "Answer the first focused question."},
        "first-follow-up-question",
    )
    second = await service.execute_create_delegation_task_action(
        db_session,
        run.id,
        {**base, "scope": "Answer a different focused question."},
        "first-follow-up-question",
    )

    followups = [
        task
        for task in await _tasks_for_run(db_session, run.id)
        if service._task_work_function(task) == "follow_up"
    ]
    assert first.id != second.id
    assert len(followups) == 2


async def test_follow_up_rejects_wrong_agent_and_parent_run(db_session, test_project):
    service, run, parent, agent, report_session = await _completed_parent_task(db_session, test_project)
    other_agent = _agent("other-follow-up", ["implementation"])
    db_session.add(other_agent)
    await db_session.flush()
    request = {
        "action_type": "create_delegation_task",
        "agent_id": str(other_agent.id),
        "work_function": "follow_up",
        "scope": "Answer the parent question.",
        "deliverable": "A short answer.",
        "parent_task_id": str(parent.id),
        "source_session_id": str(report_session.id),
    }

    with pytest.raises(HTTPException, match="Follow-up must reuse the parent's agent"):
        await service.execute_create_delegation_task_action(
            db_session, run.id, request, "wrong-agent-follow-up"
        )

    request["agent_id"] = str(agent.id)
    await service.execute_create_delegation_task_action(
        db_session, run.id, request, "valid-follow-up"
    )
    request["agent_id"] = str(other_agent.id)
    with pytest.raises(HTTPException, match="Follow-up must reuse the parent's agent"):
        await service.execute_create_delegation_task_action(
            db_session, run.id, request, "replayed-wrong-agent-follow-up"
        )

    stale_session = Session(
        task_id=parent.id,
        agent_id=agent.id,
        project_id=test_project.id,
        adapter_type="api",
        status="completed",
    )
    db_session.add(stale_session)
    await db_session.flush()
    request["agent_id"] = str(agent.id)
    request["source_session_id"] = str(stale_session.id)
    with pytest.raises(HTTPException, match="Follow-up source session was not the consumed report"):
        await service.execute_create_delegation_task_action(
            db_session, run.id, request, "wrong-report-follow-up"
        )

    _, other_run, _, _, _ = await _completed_parent_task(db_session, test_project)
    request["agent_id"] = str(agent.id)
    with pytest.raises(HTTPException, match="Follow-up parent not in this run"):
        await service.execute_create_delegation_task_action(
            db_session, other_run.id, request, "wrong-run-follow-up"
        )


async def test_tick_replay_deduplicates_regenerated_follow_up_decisions(
    db_session, test_project, stub_decision, monkeypatch
):
    # Cross-tick replay of one decision per tick; pin single-action ticks so the
    # in-tick loop does not add a second decision per tick.
    monkeypatch.setattr(settings, "orchestration_max_actions_per_tick", 1)
    service, run, parent, agent, report_session = await _completed_parent_task(db_session, test_project)
    await _seed_accepted_plan(
        db_session,
        test_project,
        service,
        run,
        agent,
        plan_items=[
            {
                "id": "already-released-item",
                "work_function": "implementation",
                "scope": "A previously released item.",
                "deliverable": "No duplicate work should release.",
                "agent_id": str(agent.id),
            }
        ],
    )
    await service.reserve_action(
        db_session,
        run_id=run.id,
        idempotency_key=f"run:{run.id}:kind:release_item:already-released-item",
        action_type="release_item",
        request={"plan_item_id": "already-released-item"},
    )
    decision_calls = 0

    def follow_up_decision(_ctx):
        nonlocal decision_calls
        decision_calls += 1
        decision = {
            "action_type": "create_delegation_task",
            "agent_id": str(agent.id),
            "work_function": "follow_up",
            "scope": "Answer the focused question about the delivered work.",
            "deliverable": "A short answer.",
            "parent_task_id": str(parent.id),
        }
        if decision_calls > 1:
            decision["source_session_id"] = str(report_session.id)
        return decision

    stub_decision(follow_up_decision)

    actions_before = await _run_actions(db_session, run.id)
    first_tick = await service.tick(db_session, run.id)
    actions_after_first = await _run_actions(db_session, run.id)
    second_tick = await service.tick(db_session, run.id)
    actions_after_second = await _run_actions(db_session, run.id)

    decisions = list(
        (await db_session.scalars(
            select(OrchestrationDecision).where(OrchestrationDecision.run_id == run.id)
        )).all()
    )
    followup_actions = [
        action for action in actions_after_second
        if action.action_type == "create_delegation_task"
        and action.request.get("work_function") == "follow_up"
    ]
    followups = [task for task in await _tasks_for_run(db_session, run.id)
                 if service._task_work_function(task) == "follow_up"]

    assert first_tick["authorized_execution"]["step"] == "next_action"
    assert second_tick["authorized_execution"]["step"] == "next_action"
    assert len(actions_after_first) - len(actions_before) <= 1
    assert len(actions_after_second) - len(actions_after_first) <= 1
    assert len({decision.id for decision in decisions}) == 2
    assert len(followup_actions) == 1
    assert len(followups) == 1


async def test_authorized_run_requests_a_plan(db_session, test_project, stub_decision):
    planner = _agent("planner", ["planning"])
    db_session.add(planner)
    await db_session.flush()

    service, goal, run = await _authorized_run(db_session, test_project)

    stub_decision(
        lambda ctx: {
            "action_type": "request_plan",
            "agent_id": _planner_agent_id(ctx),
            "scope": "Produce a plan for the objective",
            "work_function": "planning",
        }
    )

    result = await service.tick(db_session, run.id)

    await db_session.refresh(run)
    actions = await _run_actions(db_session, run.id)
    assert any(a.action_type == "request_plan" for a in actions)
    assert run.plan_state["status"] == "requested"
    assert result["authorized_execution"]["step"] == "plan_decision"


async def test_no_dispatchable_action_leaves_explicit_wait_not_stall(
    db_session, test_project, stub_decision
):
    planner = _agent("planner", ["planning"])
    db_session.add(planner)
    await db_session.flush()

    service, goal, run = await _authorized_run(db_session, test_project)
    # Force accepted-plan state with one expandable item. The first tick
    # releases that item (deterministic work); once nothing remains to
    # release, the next tick has no deterministic action left and must ask.
    await _seed_accepted_plan(db_session, test_project, service, run, planner)
    first_tick = await service.tick(db_session, run.id)
    assert first_tick["authorized_execution"]["step"] == "release_work"

    stub_decision(lambda ctx: {"action_type": "ask_human", "question": "What next?"})

    result = await service.tick(db_session, run.id)

    await db_session.refresh(run)
    actions = await _run_actions(db_session, run.id)
    # An explicit ask_human action exists; the run did not silently complete or stall.
    assert any(a.action_type == "ask_human" for a in actions)
    assert run.status != "completed"
    assert result["authorized_execution"]["step"] == "next_action"


@pytest.mark.parametrize(
    "decision",
    [
        {"action_type": "not_a_runtime_action", "reason": "Rejected by validation."},
        {"action_type": "complete_run", "reason": "Not dispatchable during execution."},
    ],
)
async def test_runtime_non_dispatchable_decision_persists_one_replayable_wait(
    db_session, test_project, stub_decision, decision
):
    agent = _agent("non-dispatchable", ["planning"])
    db_session.add(agent)
    await db_session.flush()
    service, _, run = await _authorized_run(db_session, test_project)
    await _seed_accepted_plan(db_session, test_project, service, run, agent)
    await service.tick(db_session, run.id)  # release the sole deterministic item
    actions_before = len(await _run_actions(db_session, run.id))

    stub_decision(lambda _ctx: decision)
    first = await service.tick(db_session, run.id)
    first_actions = await _run_actions(db_session, run.id)
    second = await service.tick(db_session, run.id)
    second_actions = await _run_actions(db_session, run.id)
    waits = [action for action in second_actions if action.action_type == "noop"]
    wait_events = list((await db_session.scalars(select(EventLog).where(
        EventLog.project_id == test_project.id,
        EventLog.event_type == "orchestration.waiting",
    ))).all())

    assert first["authorized_execution"]["action_id"] is not None
    # The fallback noop now creates an orchestrator backstop wait, so the next tick waits.
    assert second["authorized_execution"]["step"] == "local_liveness"
    assert second["authorized_execution"]["outcome"] == "waiting"
    assert "action_id" not in second["authorized_execution"]
    assert len(first_actions) == actions_before + 1
    assert len(second_actions) == len(first_actions)
    assert len(waits) == len(wait_events) == 1
    assert waits[0].status == "completed"


async def test_committed_regenerated_decisions_replay_once_across_fresh_services(
    db_session, test_engine, test_project
):
    """Exact decisions replay their durable action, not their new decision id."""
    agent = _agent("replay-runtime", ["implementation", "planning"])
    db_session.add(agent)
    await db_session.flush()
    service, _, run = await _authorized_run(db_session, test_project)
    await _seed_accepted_plan(db_session, test_project, service, run, agent)
    await service.tick(db_session, run.id)
    source_task = next(
        task for task in await _tasks_for_run(db_session, run.id)
        if "plan_item_gate_id" in task.metadata_.get("orchestration", {})
    )
    gate_id = source_task.metadata_["orchestration"]["plan_item_gate_id"]

    async def dispatch(db, parsed):
        decision = await OrchestrationService().record_validated_decision(
            db,
            run_id=run.id,
            input_snapshot={"replay": str(uuid.uuid4())},
            llm_output={},
            parsed_decision=parsed,
        )
        assert decision.validator_status == "accepted"
        loaded_run = await db.get(OrchestrationRun, run.id)
        return await OrchestrationDecisionDispatcher(OrchestrationService()).dispatch(
            db, loaded_run, decision
        )

    delegation = {
        "action_type": "create_delegation_task",
        "agent_id": str(agent.id),
        "work_function": "implementation",
        "scope": "Implement the replayable change.",
        "deliverable": "A tested change.",
        "reason": "First wording.",
    }
    meeting = {
        "action_type": "schedule_meeting",
        "participant_agent_ids": [str(agent.id)],
        "organizer_agent_id": str(agent.id),
        "task_id": str(source_task.id),
        "gate_id": gate_id,
        "topic": "Resolve the source-task question.",
        "reason": "First wording.",
    }
    ask = {"action_type": "ask_human", "question": "Choose the delivery scope.", "reason": "First wording."}
    noop = {"action_type": "noop", "reason": "Waiting for the next event.", "wake_when": NOOP_WAKE_WHEN}
    for parsed in (delegation, meeting, ask, noop):
        await dispatch(db_session, parsed)
    await db_session.commit()

    sessions = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with sessions.begin() as replay_db:
        replayed_delegation = {
            **delegation,
            "inputs": [],
            "forbidden_work": [],
            "success_evidence": [],
            "budget": {},
            "report_schema": {},
            "parent_task_id": None,
            "source_session_id": None,
        }
        for parsed in (replayed_delegation, meeting, ask, noop):
            await dispatch(replay_db, {**parsed, "reason": "A regenerated explanation."})

        actions = await _run_actions(replay_db, run.id)
        assert len([action for action in actions if action.action_type == "create_delegation_task"]) == 2
        assert len([action for action in actions if action.action_type == "schedule_meeting"]) == 1
        assert len([action for action in actions if action.action_type == "ask_human"]) == 1
        assert len([action for action in actions if action.action_type == "noop"]) == 1
        tasks = await _tasks_for_run(replay_db, run.id)
        assert len([task for task in tasks if service._task_work_function(task) == "implementation"]) == 1
        meetings = list((await replay_db.scalars(select(Meeting).where(Meeting.source_task_id == source_task.id))).all())
        assert len(meetings) == 1
        event_counts = {
            "orchestration.delegation_task_created": 2,
            "orchestration.meeting_scheduled": 1,
            "orchestration.human_input_required": 1,
            "orchestration.waiting": 1,
        }
        for event_type, expected_count in event_counts.items():
            assert len(list((await replay_db.scalars(select(EventLog).where(
                EventLog.project_id == test_project.id, EventLog.event_type == event_type
            ))).all())) == expected_count

        await dispatch(replay_db, {**ask, "question": "Choose the delivery budget.", "reason": "New request."})
        asks = [action for action in await _run_actions(replay_db, run.id) if action.action_type == "ask_human"]
        assert len(asks) == 2


async def test_two_task_cap_releases_at_most_two(db_session, test_project, stub_decision):
    """3 independent (no depends_on) ready plan items -- only 2 may be
    in-flight at once. See OrchestrationService._release_ready_work."""
    planner = _agent("planner", ["implementation"])
    db_session.add(planner)
    await db_session.flush()

    service, goal, run = await _authorized_run(db_session, test_project)
    items = [
        {
            "id": f"item-{n}",
            "work_function": "implementation",
            "scope": f"Independent unit of work {n}.",
            "deliverable": f"Output for item {n}.",
            "agent_id": str(planner.id),
        }
        for n in range(1, 4)
    ]
    await _seed_accepted_plan(db_session, test_project, service, run, planner, plan_items=items)

    result = await service.tick(db_session, run.id)
    assert result["authorized_execution"]["step"] == "release_work"
    assert result["authorized_execution"]["released"] == 2

    tasks = await _tasks_for_run(db_session, run.id)
    in_flight = [
        t for t in tasks
        if t.status in ("backlog", "ready", "in_progress", "blocked")
        and OrchestrationService._task_work_function(t) not in ("planning", "summarization")
    ]
    assert len(in_flight) == 2

    # The third item releases only once one of the first two terminalizes.
    in_flight[0].status = "done"
    await db_session.flush()
    second_result = await service.tick(db_session, run.id)
    assert second_result["authorized_execution"]["step"] == "release_work"
    assert second_result["authorized_execution"]["released"] == 1

    tasks = await _tasks_for_run(db_session, run.id)
    in_flight_after = [
        t for t in tasks
        if t.status in ("backlog", "ready", "in_progress", "blocked")
        and OrchestrationService._task_work_function(t) not in ("planning", "summarization")
    ]
    assert len(in_flight_after) == 2


def _plan_item_ids(tasks):
    return {t.metadata_["orchestration_plan_item"]["id"] for t in tasks}


async def test_dependent_item_waits_for_accepted_predecessor_gate(
    db_session, test_project, stub_decision
):
    """item-b depends_on "item-a" (bare id) and item-c depends_on
    "plan_item:item-a" (gate-key form) -- both must wait until item-a's
    plan_item gate is accepted. See _item_dependencies_accepted."""
    planner = _agent("planner", ["implementation"])
    verifier = _agent("dependency-verifier", ["validation"])
    planner.role, verifier.role = "developer", "validator"
    db_session.add_all([planner, verifier])
    await db_session.flush()

    service, goal, run = await _authorized_run(db_session, test_project)
    items = [
        {
            "id": "item-a",
            "work_function": "implementation",
            "scope": "Independent predecessor unit of work.",
            "deliverable": "Output for item-a.",
            "agent_id": str(planner.id),
        },
        {
            "id": "item-b",
            "work_function": "implementation",
            "scope": "Depends on item-a (bare id form).",
            "deliverable": "Output for item-b.",
            "agent_id": str(planner.id),
            "depends_on": ["item-a"],
        },
        {
            "id": "item-c",
            "work_function": "implementation",
            "scope": "Depends on item-a (gate-key form).",
            "deliverable": "Output for item-c.",
            "agent_id": str(planner.id),
            "depends_on": ["plan_item:item-a"],
        },
    ]
    await _seed_accepted_plan(db_session, test_project, service, run, planner, plan_items=items)

    result = await service.tick(db_session, run.id)
    assert result["authorized_execution"]["step"] == "release_work"
    assert result["authorized_execution"]["released"] == 1

    tasks = await _tasks_for_run(db_session, run.id)
    in_flight = [
        t for t in tasks
        if t.status in ("backlog", "ready", "in_progress", "blocked")
        and OrchestrationService._task_work_function(t) not in ("planning", "summarization")
    ]
    # Only the independent predecessor released; both dependents are still gated.
    assert _plan_item_ids(in_flight) == {"item-a"}
    item_a_task = in_flight[0]

    # Bound verification accepts the predecessor gate and frees its slot.
    gate = (
        await db_session.execute(
            select(OrchestrationGate).where(
                OrchestrationGate.run_id == run.id,
                OrchestrationGate.success_criterion_key == "plan_item:item-a",
            )
        )
    ).scalar_one()
    await _complete_task_session(
        db_session, test_project.id, item_a_task, planner.id,
        json.dumps({"status": "done", "changes": ["Output for item-a."]}),
    )
    action = await service.execute_request_verification_action(
        db_session,
        run.id,
        {"action_type": "request_verification", "gate_id": str(gate.id), "work_function": "validation"},
        f"run:{run.id}:kind:dependency-verification",
    )
    verification_task = await db_session.get(Task, action.target_id)
    assert verification_task is not None
    await _complete_task_session(
        db_session, test_project.id, verification_task, verifier.id,
        json.dumps({"status": "done", "verdict": "accepted", "evidence": ["Independent proof."]}),
    )
    events = await service._new_events(db_session, test_project.id, run.event_cursor)
    await service._ingest_evidence_from_events(db_session, run, events)
    await service.validate_open_gates(db_session, run.id)
    await db_session.refresh(gate)
    assert gate.status == "accepted"

    second_result = await service.tick(db_session, run.id)
    assert second_result["authorized_execution"]["step"] == "release_work"
    assert second_result["authorized_execution"]["released"] == 2

    tasks = await _tasks_for_run(db_session, run.id)
    in_flight_after = [
        t for t in tasks
        if t.status in ("backlog", "ready", "in_progress", "blocked")
        and OrchestrationService._task_work_function(t) not in ("planning", "summarization")
    ]
    assert _plan_item_ids(in_flight_after) == {"item-b", "item-c"}


async def test_cancel_terminates_active_session_and_ignores_late_output(
    db_session, test_project, stub_decision
):
    planner = _agent("cancel", ["implementation"])
    db_session.add(planner)
    await db_session.flush()

    service, goal, run = await _authorized_run(db_session, test_project)
    await _seed_accepted_plan(db_session, test_project, service, run, planner)
    await service.tick(db_session, run.id)
    task = (await _tasks_for_run(db_session, run.id))[0]
    gate = (
        await db_session.execute(
            select(OrchestrationGate).where(
                OrchestrationGate.run_id == run.id,
                OrchestrationGate.success_criterion_key == "plan_item:item-1",
            )
        )
    ).scalar_one()
    assert gate.status == "open"
    _, session_id = await TaskService().run(db_session, test_project.id, task.id)
    session = await db_session.get(Session, session_id)
    assert session is not None and session.status == "pending"

    cancelled_goal, cancelled_run = await service.cancel_goal(db_session, test_project.id, goal.id)

    assert cancelled_goal.status == "cancelled"
    assert cancelled_run is not None and cancelled_run.status == "cancelled"
    assert task.status == "cancelled"
    assert session.status == "cancelled"

    await emit_event(
        db_session,
        test_project.id,
        "session.completed",
        {"session_id": str(session.id), "task_id": str(task.id)},
    )
    await service.tick(db_session, run.id)
    await db_session.refresh(gate)
    assert gate.status == "open"


async def test_cancel_only_stops_owned_runner_sessions(db_session, test_project, monkeypatch):
    agent = _agent("cancel-owned", ["implementation"])
    db_session.add(agent)
    await db_session.flush()

    async def active_work(run):
        await _seed_accepted_plan(
            db_session, test_project, service, run, agent,
            plan_items=[{
                "id": "owned-work",
                "work_function": "implementation",
                "scope": "Perform cancellable work.",
                "deliverable": "A cancellable result.",
                "agent_id": str(agent.id),
            }],
        )
        await service.tick(db_session, run.id)
        task = next(
            task for task in await _tasks_for_run(db_session, run.id)
            if service._task_work_function(task) == "implementation"
        )
        _, session_id = await TaskService().run(db_session, test_project.id, task.id)
        session = await db_session.get(Session, session_id)
        assert session is not None
        session.status = "running"
        session.runner_task_id = f"runner:{run.id}"
        await db_session.flush()
        return task, session

    service, goal, run = await _authorized_run(db_session, test_project)
    owned_task, owned_session = await active_work(run)
    _, _, other_run = await _authorized_run(db_session, test_project)
    other_task, other_session = await active_work(other_run)
    cancelled_runner_ids = []

    async def cancel_runner(task_id):
        cancelled_runner_ids.append(task_id)

    monkeypatch.setattr("huddleroom.workers.task_runner.cancel_task", cancel_runner)

    await service.cancel_goal(db_session, test_project.id, goal.id)

    assert owned_task.status == owned_session.status == "cancelled"
    assert cancelled_runner_ids == [f"runner:{run.id}"]
    assert other_task.status == "in_progress"
    assert other_session.status == "running"
    events = list(await db_session.scalars(
        select(EventLog).where(EventLog.event_type.in_(("task.status_changed", "session.cancelled")))
    ))
    assert any(event.event_type == "session.cancelled" and event.payload["session_id"] == str(owned_session.id)
               for event in events)
    assert any(event.event_type == "task.status_changed" and event.payload["task_id"] == str(owned_task.id)
               and event.payload["status"] == "cancelled" for event in events)


async def test_outcome_goal_golden_path_reaches_completed_phase(
    db_session, test_project, stub_decision, safe_effectiveness_review_continue, monkeypatch
):
    """A real Outcome run only completes after independent evidence and closeout.

    Removing Start, accepting the producer session, or forgetting the terminal
    phase transition each makes this contract fail.
    """
    # Step-by-step golden path; pin single-action ticks so each tick asserts one step.
    monkeypatch.setattr(settings, "orchestration_max_actions_per_tick", 1)
    planner = _agent("golden-planner", ["planning"])
    producer = _agent("golden-producer", ["implementation"])
    verifier = _agent("golden-verifier", ["validation"])
    summarizer = _agent("golden-summarizer", ["summarization"])
    planner.role = "planner"
    producer.role = "developer"
    verifier.role = "validator"
    summarizer.role = "technical writer"
    db_session.add_all([planner, producer, verifier, summarizer])
    await db_session.flush()

    service, goal, run = await _authorized_run(db_session, test_project)
    goal.weight = "trivial"
    await heal_baseline_drift_for_test(db_session, goal, run)
    stub_decision(
        lambda ctx: {
            "action_type": "request_plan",
            "agent_id": _planner_agent_id(ctx),
            "work_function": "planning",
            "scope": "Plan one independently verifiable implementation item.",
        }
    )
    first_tick = await service.tick(db_session, run.id)
    assert first_tick["authorized_execution"]["step"] == "plan_decision"
    plan_request = next(
        action for action in await _run_actions(db_session, run.id)
        if action.action_type == "request_plan"
    )
    planning_task = await db_session.get(Task, plan_request.target_id)
    await _complete_task_session(
        db_session,
        test_project.id,
        planning_task,
        planner.id,
        json.dumps({"status": "done", "changes": ["Prepared the execution plan."]}),
    )
    artifact = await ArtifactService().create(
        db_session,
        project_id=test_project.id,
        name="golden-runtime-plan",
        artifact_type="plan",
        linked_task_id=planning_task.id,
        created_by_agent=planner.id,
        metadata={
            "kind": "implementation_plan",
            "plan_items": [{
                "id": "deliver",
                "work_function": "implementation",
                "scope": "Deliver one independently verifiable change.",
                "deliverable": "A tested implementation result.",
                "agent_id": str(producer.id),
                "required_evidence": {
                    "required_source_types": ["task", "review"],
                    "min_count": 2,
                    "requires_independent_agent": True,
                    "work_producer_agent_id": str(producer.id),
                },
            }],
        },
    )
    stub_decision(
        lambda _ctx: {"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)}
    )
    accept_tick = await service.tick(db_session, run.id)
    assert accept_tick["authorized_execution"]["step"] == "plan_decision"
    await db_session.refresh(run)
    assert run.plan_state["status"] == "accepted"
    assert any(action.action_type == "accept_plan" for action in await _run_actions(db_session, run.id))
    release_tick = await service.tick(db_session, run.id)
    assert release_tick["authorized_execution"]["step"] == "release_work"
    work_task = next(
        task for task in await _tasks_for_run(db_session, run.id)
        if service._task_work_function(task) == "implementation"
    )
    work_gate_id = uuid.UUID(work_task.metadata_["orchestration"]["plan_item_gate_id"])
    work_gate = await db_session.get(OrchestrationGate, work_gate_id)
    assert work_gate.required_evidence["success_criterion_keys"] == ["done"]

    producer_session = await _complete_task_session(
        db_session,
        test_project.id,
        work_task,
        producer.id,
        json.dumps({"status": "done", "changes": ["Implemented the requested change."]}),
    )
    stub_decision(
        lambda _ctx: {
            "action_type": "request_verification",
            "gate_id": str(work_gate.id),
            "work_function": "validation",
        }
    )
    await service.tick(db_session, run.id)
    await db_session.refresh(work_gate)
    assert work_gate.status != "accepted", "the producing agent cannot accept its own gate"

    verification = next(
        action for action in await _run_actions(db_session, run.id)
        if action.action_type == "request_verification"
    )
    verification_task = await db_session.get(Task, verification.target_id)
    assert verification_task is not None and verification_task.assigned_to == verifier.id
    binding = verification_task.metadata_["orchestration"]
    assert binding["verification_action_id"] == str(verification.id)
    assert binding["verification_gate_id"] == str(work_gate.id)
    assert binding["source_task_id"] == str(work_task.id)
    assert binding["producer_agent_id"] == str(producer.id)
    assert binding["verifier_agent_id"] == str(verifier.id)
    verifier_session = await _complete_task_session(
        db_session,
        test_project.id,
        verification_task,
        verifier.id,
        json.dumps({"status": "done", "verdict": "accepted", "evidence": ["Independently validated the change."]}),
    )
    stub_decision(lambda _ctx: {"action_type": "noop"})
    await service.tick(db_session, run.id)
    await db_session.refresh(work_gate)
    assert work_gate.status == "accepted", work_gate.failure_reason

    actions = await _run_actions(db_session, run.id)
    summary_request = next((action for action in actions if action.action_type == "request_final_summary"), None)
    gates = list((await db_session.scalars(select(OrchestrationGate).where(OrchestrationGate.run_id == run.id))).all())
    assert summary_request is not None, ([(gate.success_criterion_key, gate.status) for gate in gates], run.active_blockers)
    summary_task = await db_session.get(Task, summary_request.target_id)
    accepted_evidence = list((await db_session.scalars(
        select(OrchestrationEvidence.id).where(
            OrchestrationEvidence.run_id == run.id,
            OrchestrationEvidence.gate_id == work_gate.id,
            OrchestrationEvidence.verdict == "accepted",
            OrchestrationEvidence.source_type == "verification",
        )
    )).all())
    assert accepted_evidence
    await _complete_task_session(
        db_session,
        test_project.id,
        summary_task,
        summarizer.id,
        json.dumps({
            "summary": "The independent verifier accepted the delivered work.",
            "criteria": [{"criterion_key": "done", "evidence_ids": [str(accepted_evidence[0])]}],
            "unresolved_gaps": [],
        }),
    )
    await service.tick(db_session, run.id)

    await db_session.refresh(goal)
    await db_session.refresh(run)
    assert producer_session.agent_id != verifier_session.agent_id
    assert goal.status == "completed"
    assert run.status == "completed"
    assert run.phase == "completed"


@pytest.mark.parametrize("reuse_first_criterion_evidence", [False, True])
async def test_outcome_two_criterion_plan_requires_criterion_scoped_verification_evidence(
    db_session,
    test_project,
    stub_decision,
    safe_agent_definition_review,
    safe_effectiveness_review_continue,
    reuse_first_criterion_evidence,
    monkeypatch,
):
    """Real two-criterion Outcome work reaches completion only with linked verifier proof."""
    # Pin single-action ticks: with the loop, the accept tick also releases work, and the
    # following tick would fall through to a stub that still returns accept_plan (409).
    monkeypatch.setattr(settings, "orchestration_max_actions_per_tick", 1)
    planner = _agent("two-criterion-planner", ["planning"])
    producer_a = _agent("two-criterion-producer-a", ["implementation"])
    producer_b = _agent("two-criterion-producer-b", ["implementation"])
    verifier = _agent("two-criterion-verifier", ["validation"])
    summarizer = _agent("two-criterion-summarizer", ["summarization"])
    planner.role = "planner"
    producer_a.role = producer_b.role = "developer"
    verifier.role = "validator"
    summarizer.role = "technical writer"
    db_session.add_all([planner, producer_a, producer_b, verifier, summarizer])
    await db_session.flush()

    service, goal, run = await _authorized_run(
        db_session,
        test_project,
        success_criteria=[
            {"key": "implemented", "description": "The implementation is delivered."},
            {"key": "validated", "description": "The validation is delivered."},
        ],
    )
    goal.weight = "trivial"
    await heal_baseline_drift_for_test(db_session, goal, run)
    stub_decision(
        lambda ctx: {
            "action_type": "request_plan",
            "agent_id": _planner_agent_id(ctx),
            "work_function": "planning",
            "scope": "Plan two independently verified criterion-linked items.",
        }
    )
    await service.tick(db_session, run.id)
    plan_request = next(
        action for action in await _run_actions(db_session, run.id)
        if action.action_type == "request_plan"
    )
    planning_task = await db_session.get(Task, plan_request.target_id)
    assert planning_task is not None
    await _complete_task_session(
        db_session, test_project.id, planning_task, planner.id,
        json.dumps({"status": "done", "changes": ["Prepared a linked two-item plan."]}),
    )
    artifact = await ArtifactService().create(
        db_session,
        project_id=test_project.id,
        name="two-criterion-runtime-plan",
        artifact_type="plan",
        linked_task_id=planning_task.id,
        created_by_agent=planner.id,
        metadata={
            "kind": "implementation_plan",
            "plan_items": [
                {
                    "id": "implement",
                    "work_function": "implementation",
                    "scope": "Deliver the implementation.",
                    "deliverable": "An independently verified implementation.",
                    "agent_id": str(producer_a.id),
                    "success_criterion_keys": ["implemented"],
                },
                {
                    "id": "validate",
                    "work_function": "implementation",
                    "scope": "Deliver the validation result.",
                    "deliverable": "An independently verified validation.",
                    "agent_id": str(producer_b.id),
                    "success_criterion_keys": ["validated"],
                },
            ],
        },
    )
    stub_decision(lambda _ctx: {"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)})
    await service.tick(db_session, run.id)
    await db_session.refresh(run)
    assert run.plan_state["status"] == "accepted"
    await service.tick(db_session, run.id)

    work_tasks = {
        task.metadata_["orchestration"]["plan_item_id"]: task
        for task in await _tasks_for_run(db_session, run.id)
        if service._task_work_function(task) == "implementation"
    }
    assert set(work_tasks) == {"implement", "validate"}
    gates = {
        item_id: await db_session.get(
            OrchestrationGate,
            uuid.UUID(task.metadata_["orchestration"]["plan_item_gate_id"]),
        )
        for item_id, task in work_tasks.items()
    }
    assert all(gate is not None for gate in gates.values())
    assert gates["implement"].required_evidence["success_criterion_keys"] == ["implemented"]
    assert gates["validate"].required_evidence["success_criterion_keys"] == ["validated"]

    await _complete_task_session(
        db_session, test_project.id, work_tasks["implement"], producer_a.id,
        json.dumps({"status": "done", "changes": ["Delivered implementation."]}),
    )
    await _complete_task_session(
        db_session, test_project.id, work_tasks["validate"], producer_b.id,
        json.dumps({"status": "done", "changes": ["Delivered validation."]}),
    )
    for item_id in ("implement", "validate"):
        stub_decision(
            lambda _ctx, item_id=item_id: {
                "action_type": "request_verification",
                "gate_id": str(gates[item_id].id),
                "work_function": "validation",
            }
        )
        await service.tick(db_session, run.id)
    verification_actions = {
        action.request["gate_id"]: action
        for action in await _run_actions(db_session, run.id)
        if action.action_type == "request_verification"
    }
    assert set(verification_actions) == {str(gates["implement"].id), str(gates["validate"].id)}
    verification_tasks = {
        item_id: await db_session.get(
            Task, verification_actions[str(gate.id)].target_id
        )
        for item_id, gate in gates.items()
    }
    assert all(task is not None and task.assigned_to == verifier.id for task in verification_tasks.values())
    for item_id, task in verification_tasks.items():
        binding = task.metadata_["orchestration"]
        assert binding["verification_gate_id"] == str(gates[item_id].id)
        assert binding["producer_agent_id"] == str(
            producer_a.id if item_id == "implement" else producer_b.id
        )
        await _complete_task_session(
            db_session,
            test_project.id,
            task,
            verifier.id,
            json.dumps({
                "status": "done",
                "verdict": "accepted",
                "evidence": [f"Independent verification for {item_id}."],
            }),
        )
    stub_decision(lambda _ctx: {"action_type": "noop"})
    await service.tick(db_session, run.id)
    await db_session.refresh(gates["implement"])
    await db_session.refresh(gates["validate"])
    assert all(gate.status == "accepted" for gate in gates.values())

    summary_action = next(
        action for action in await _run_actions(db_session, run.id)
        if action.action_type == "request_final_summary"
    )
    summary_task = await db_session.get(Task, summary_action.target_id)
    assert summary_task is not None and summary_task.assigned_to == summarizer.id
    summary_delegation = await db_session.scalar(
        select(OrchestrationAction).where(
            OrchestrationAction.run_id == run.id,
            OrchestrationAction.action_type == "create_delegation_task",
            OrchestrationAction.target_id == summary_task.id,
        )
    )
    manifest = json.loads(next(
        value.removeprefix("Criterion-scoped accepted verification evidence: ")
        for value in summary_delegation.request["inputs"]
        if value.startswith("Criterion-scoped accepted verification evidence: ")
    ))
    evidence_ids = {
        item_id: str(await db_session.scalar(
            select(OrchestrationEvidence.id).where(
                OrchestrationEvidence.run_id == run.id,
                OrchestrationEvidence.gate_id == gate.id,
                OrchestrationEvidence.source_type == "verification",
                OrchestrationEvidence.verdict == "accepted",
            )
        ))
        for item_id, gate in gates.items()
    }
    assert manifest == {
        "implemented": [evidence_ids["implement"]],
        "validated": [evidence_ids["validate"]],
    }
    validation_evidence = evidence_ids["implement"] if reuse_first_criterion_evidence else evidence_ids["validate"]
    await _complete_task_session(
        db_session,
        test_project.id,
        summary_task,
        summarizer.id,
        json.dumps({
            "summary": "Both linked work items were independently verified.",
            "criteria": [
                {"criterion_key": "implemented", "evidence_ids": [evidence_ids["implement"]]},
                {"criterion_key": "validated", "evidence_ids": [validation_evidence]},
            ],
            "unresolved_gaps": [],
        }),
    )
    result = await service.tick(db_session, run.id)
    await db_session.refresh(goal)
    await db_session.refresh(run)
    if reuse_first_criterion_evidence:
        assert result["run_completed"] is False
        assert goal.status != "completed" and run.status != "completed"
    else:
        assert result["run_completed"] is True
        assert goal.status == run.status == "completed"
        assert run.phase == "completed"


async def test_legacy_accepted_multi_criterion_plan_blocks_once_without_releasing_work(
    db_session,
    test_engine,
    test_project,
    safe_effectiveness_review_continue,
    monkeypatch,
    stub_decision,
):
    """An accepted legacy multi-criterion plan without links fails closed after restart."""
    planner = _agent("legacy-multi-planner", ["planning"])
    producer = _agent("legacy-multi-producer", ["implementation"])
    planner.role, producer.role = "planner", "developer"

    from huddleroom.services.orchestration_agent_definition_analyzer import (
        AgentDefinitionSemanticAnalyzer,
        SemanticAgentAssessment,
    )

    async def approve_agent_definition(
        _self, _agent_snapshot, _goal_snapshot, candidate_work_functions, project=None, *, project_id=None
    ):
        return SemanticAgentAssessment("approved", (), "test fixture", tuple(candidate_work_functions))

    async def approve_agent_definition_request(
        _self, _request, candidate_work_functions, *, project_id=None
    ):
        return SemanticAgentAssessment("approved", (), "test fixture", tuple(candidate_work_functions))

    monkeypatch.setattr(AgentDefinitionSemanticAnalyzer, "review", approve_agent_definition)
    monkeypatch.setattr(AgentDefinitionSemanticAnalyzer, "review_request", approve_agent_definition_request)
    db_session.add_all([planner, producer])
    await db_session.flush()
    service, goal, run = await _authorized_run(
        db_session,
        test_project,
        success_criteria=[
            {"key": "implemented", "description": "Implementation is delivered."},
            {"key": "validated", "description": "Validation is delivered."},
        ],
    )
    await _seed_accepted_plan(
        db_session,
        test_project,
        service,
        run,
        planner,
        plan_items=[{
            "id": "legacy-multi",
            "work_function": "implementation",
            "scope": "A formerly accepted plan item.",
            "deliverable": "An independently verified result.",
            "agent_id": str(producer.id),
            "success_criterion_keys": ["implemented", "validated"],
        }],
    )
    artifact = await db_session.get(Artifact, uuid.UUID(run.plan_state["accepted_artifact_id"]))
    assert artifact is not None
    # Legacy accepted runs have no immutable authority snapshot and must stop.
    run.plan_state = {key: value for key, value in run.plan_state.items() if key != "accepted_plan_snapshot"}
    await db_session.commit()

    sessions = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with sessions.begin() as first_tick_db:
        persisted_artifact = await first_tick_db.get(Artifact, artifact.id)
        persisted_goal_before_tick = await first_tick_db.get(OrchestrationGoal, goal.id)
        persisted_run_before_tick = await first_tick_db.get(OrchestrationRun, run.id)
        assert persisted_artifact is not None
        assert persisted_goal_before_tick is not None and persisted_run_before_tick is not None
        assert persisted_run_before_tick.plan_state["status"] == "accepted"
        assert "accepted_plan_snapshot" not in persisted_run_before_tick.plan_state
        async def forbidden(*_args, **_kwargs):
            raise AssertionError("snapshot integrity must stop before recovery or effectiveness work")

        monkeypatch.setattr(OrchestrationService, "recover_run", forbidden)
        monkeypatch.setattr(EffectivenessReviewProcess, "advance", forbidden)
        stub_decision(lambda _ctx: pytest.fail("snapshot integrity must stop before LLM work"))
        first = await OrchestrationService().tick(first_tick_db, run.id)
        persisted_goal = await first_tick_db.get(OrchestrationGoal, goal.id)
        persisted_run = await first_tick_db.get(OrchestrationRun, run.id)
        assert persisted_goal is not None and persisted_run is not None
        assert persisted_goal.status == persisted_run.status == "blocked"
        blockers = [item for item in persisted_run.active_blockers if item["kind"] == "plan_criterion_integrity"]
        assert blockers == [{
            "kind": "plan_criterion_integrity",
            "reason": "Accepted plan snapshot is missing or unsupported",
            "recommended_action": "Replace or supersede before authorization, or request human intervention.",
        }]
        action_count = len(await _run_actions(first_tick_db, run.id))
        assert not [action for action in await _run_actions(first_tick_db, run.id)
                    if action.action_type in {"expand_plan_item", "release_item"}]
        assert not [
            task for task in await _tasks_for_run(first_tick_db, run.id)
            if OrchestrationService._task_work_function(task) == "implementation"
        ]

    async with sessions.begin() as replay_db:
        replay = await OrchestrationService().tick(replay_db, run.id)
        persisted_run = await replay_db.get(OrchestrationRun, run.id)
        assert persisted_run is not None
        assert len([item for item in persisted_run.active_blockers if item["kind"] == "plan_criterion_integrity"]) == 1
        assert len(await _run_actions(replay_db, run.id)) == action_count
        assert not [
            task for task in await _tasks_for_run(replay_db, run.id)
            if OrchestrationService._task_work_function(task) == "implementation"
        ]


async def test_accepted_snapshot_ignores_later_artifact_plan_mutation(
    db_session, test_project, safe_effectiveness_review_continue
):
    planner = _agent("snapshot-planner", ["planning"])
    db_session.add(planner)
    await db_session.flush()
    service, _goal, run = await _authorized_run(db_session, test_project)
    await _seed_accepted_plan(db_session, test_project, service, run, planner, plan_items=[{
        "id": "frozen-item", "work_function": "planning", "scope": "Frozen scope",
        "deliverable": "Frozen deliverable", "agent_id": str(planner.id),
    }])
    artifact = await db_session.get(Artifact, uuid.UUID(run.plan_state["accepted_artifact_id"]))
    assert artifact is not None
    artifact.metadata_ = {**artifact.metadata_, "plan_items": [{
        "id": "injected-item", "work_function": "planning", "scope": "Injected scope",
        "deliverable": "Injected deliverable", "agent_id": str(planner.id),
    }]}
    await service._release_ready_work(db_session, await db_session.get(OrchestrationGoal, run.goal_id), run)
    tasks = await _tasks_for_run(db_session, run.id)
    task = next(
        task for task in tasks
        if task.metadata_.get("orchestration", {}).get("plan_item_id") == "frozen-item"
    )
    item = task.metadata_["orchestration_plan_item"]
    assert item["id"] == "frozen-item"
    assert item["scope"] == "Frozen scope"


@pytest.mark.parametrize("graph_bound", [False, True])
async def test_outcome_plan_item_gate_requires_authoritative_producer_and_independent_verification(
    db_session, test_project, stub_decision, safe_effectiveness_review_continue, graph_bound
):
    """Planner metadata cannot weaken Outcome work's producer/reviewer split."""
    producer = _agent("defaulted-producer", ["implementation"])
    verifier = _agent("defaulted-verifier", ["validation"])
    producer.role = "developer"
    verifier.role = "validator"
    db_session.add_all([producer, verifier])
    await db_session.flush()
    service, _goal, run = await _authorized_run(db_session, test_project)
    await _seed_accepted_plan(
        db_session,
        test_project,
        service,
        run,
        producer,
        plan_items=[{
            "id": "planner-weakened-gate",
            "work_function": "implementation",
            "scope": "Deliver work with deliberately weak planner evidence metadata.",
            "deliverable": "A reviewed result.",
            "agent_id": str(producer.id),
            "required_evidence": {"required_source_types": ["task"], "min_count": 1, "requires_independent_agent": False},
        }],
    )
    await service.tick(db_session, run.id)
    work_task = next(task for task in await _tasks_for_run(db_session, run.id)
                     if service._task_work_function(task) == "implementation")
    gate = await db_session.get(
        OrchestrationGate, uuid.UUID(work_task.metadata_["orchestration"]["plan_item_gate_id"])
    )

    producer_session = await _complete_task_session(
        db_session, test_project.id, work_task, producer.id,
        json.dumps({"status": "done", "changes": ["Delivered work."]}),
    )
    stub_decision(lambda _ctx: {"action_type": "noop"})
    await service.tick(db_session, run.id)
    await db_session.refresh(gate)
    assert gate.status == "open"

    verification = await service.execute_request_verification_action(
        db_session,
        run.id,
        {"action_type": "request_verification", "gate_id": str(gate.id), "work_function": "validation"},
        f"run:{run.id}:kind:defaulted-independent-verification",
    )
    verification_task = await db_session.get(Task, verification.target_id)
    assert verification_task is not None and verification_task.assigned_to == verifier.id
    binding = verification_task.metadata_["orchestration"]
    assert binding["verification_action_id"] == str(verification.id)
    assert binding["verification_gate_id"] == str(gate.id)
    assert binding["source_task_id"] == str(work_task.id)
    assert binding["producer_agent_id"] == str(producer.id)
    assert binding["verifier_agent_id"] == str(verifier.id)
    if graph_bound:
        graph = Graph(
            project_id=test_project.id,
            name=f"verification-{uuid.uuid4()}",
            description="Bound verification test graph.",
            definition={},
            triggers=[],
        )
        db_session.add(graph)
        await db_session.flush()
        instance = GraphRun(
            graph_id=graph.id,
            project_id=test_project.id,
            linked_task_id=verification_task.id,
            current_node="done",
            status="completed",
        )
        db_session.add(instance)
        await db_session.flush()
        verification_task.graph_run_id = instance.id
    await _complete_task_session(
        db_session, test_project.id, verification_task, verifier.id,
        json.dumps({"status": "done", "verdict": "accepted", "evidence": ["Reviewed the delivered work."]}),
        graph_run_id=verification_task.graph_run_id,
    )
    await service.tick(db_session, run.id)
    await db_session.refresh(gate)
    assert gate.status == "accepted"
    evidence = list((await db_session.scalars(
        select(OrchestrationEvidence).where(OrchestrationEvidence.gate_id == gate.id)
    )).all())
    assert not any(item.source_type == "task" and item.source_id == verification_task.id for item in evidence)
    assert next(item for item in evidence if item.source_id == producer_session.id).verdict == "candidate"


@pytest.mark.parametrize("status", ["completed", "cancelled"])
async def test_reconciliation_leaves_historical_outcome_gate_acceptance_immutable(
    db_session, test_project, status
):
    goal = OrchestrationGoal(
        project_id=test_project.id,
        objective="Preserve historical acceptance.",
        original_request="Preserve historical acceptance.",
        success_criteria=[], constraints={}, budget={}, goal_type="outcome", status=status,
    )
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(
        goal_id=goal.id, status=status, phase="authorized", plan_state={"status": "accepted"}
    )
    db_session.add(run)
    await db_session.flush()
    gate = OrchestrationGate(
        run_id=run.id,
        success_criterion_key="plan_item:historical",
        gate_type="work_completed",
        required_evidence={"required_source_types": ["task"], "min_count": 1},
        status="accepted",
        accepted_at=_utcnow(),
    )
    db_session.add(gate)
    await db_session.flush()

    assert await OrchestrationService()._reconcile_outcome_gate_acceptance(db_session, goal, run) == 0
    await db_session.refresh(gate)
    assert gate.status == "accepted"


async def test_paused_run_reconciles_weak_outcome_gate_without_resuming_execution(db_session, test_project):
    goal = OrchestrationGoal(
        project_id=test_project.id,
        objective="Repair paused gate trust.",
        original_request="Repair paused gate trust.",
        success_criteria=[], constraints={}, budget={}, goal_type="outcome",
    )
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(
        goal_id=goal.id, status="paused", phase="authorized", plan_state={"status": "accepted"}
    )
    db_session.add(run)
    await db_session.flush()
    gate = OrchestrationGate(
        run_id=run.id,
        success_criterion_key="plan_item:paused",
        gate_type="work_completed",
        required_evidence={"required_source_types": ["task"], "min_count": 1},
        status="accepted", accepted_at=_utcnow(),
    )
    db_session.add(gate)
    await db_session.flush()

    await OrchestrationService().tick(db_session, run.id)
    await db_session.refresh(gate)
    await db_session.refresh(run)
    assert gate.status == "open"
    assert run.status == "paused"


async def test_reconciliation_repair_survives_restart_and_emits_one_audit_event(test_engine, tmp_path):
    sessions = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with sessions.begin() as setup_db:
        workspace = tmp_path / "restart-repair-workspace"
        workspace.mkdir()
        project = Project(name="Restart Repair", workspace_path=str(workspace), config={})
        setup_db.add(project)
        await setup_db.flush()
        goal = OrchestrationGoal(
            project_id=project.id, objective="Repair after restart.", original_request="Repair after restart.",
            success_criteria=[], constraints={}, budget={}, goal_type="outcome",
        )
        setup_db.add(goal)
        await setup_db.flush()
        run = OrchestrationRun(goal_id=goal.id, phase="authorized", plan_state={"status": "accepted"})
        setup_db.add(run)
        await setup_db.flush()
        gate = OrchestrationGate(
            run_id=run.id, success_criterion_key="plan_item:restart", gate_type="work_completed",
            required_evidence={"required_source_types": ["task"], "min_count": 1},
            status="accepted", accepted_at=_utcnow(),
        )
        setup_db.add(gate)
        await setup_db.flush()
        project_id, goal_id, run_id, gate_id = project.id, goal.id, run.id, gate.id

    async with sessions.begin() as repair_db:
        project = await repair_db.get(Project, project_id)
        goal = await repair_db.get(OrchestrationGoal, goal_id)
        run = await repair_db.get(OrchestrationRun, run_id)
        assert project is not None and goal is not None and run is not None
        assert await OrchestrationService()._reconcile_outcome_gate_acceptance(repair_db, goal, run) == 1

    async with sessions.begin() as replay_db:
        goal = await replay_db.get(OrchestrationGoal, goal_id)
        run = await replay_db.get(OrchestrationRun, run_id)
        gate = await replay_db.get(OrchestrationGate, gate_id)
        assert goal is not None and run is not None and gate is not None
        assert await OrchestrationService()._reconcile_outcome_gate_acceptance(replay_db, goal, run) == 0
        repairs = list((await replay_db.scalars(select(EventLog).where(
            EventLog.event_type == "orchestration.gate_repaired"
        ))).all())
        assert gate.status == "open"
        assert len(repairs) == 1


async def test_paused_tick_open_gate_normalization_persists_across_sessions(test_engine, tmp_path):
    sessions = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with sessions.begin() as setup_db:
        workspace = tmp_path / "paused-repair-workspace"
        workspace.mkdir()
        project = Project(name="Paused Repair", workspace_path=str(workspace), config={})
        setup_db.add(project)
        await setup_db.flush()
        goal = OrchestrationGoal(
            project_id=project.id, objective="Repair paused trust.", original_request="Repair paused trust.",
            success_criteria=[], constraints={}, budget={}, goal_type="outcome",
        )
        setup_db.add(goal)
        await setup_db.flush()
        run = OrchestrationRun(goal_id=goal.id, status="paused", phase="authorized", plan_state={"status": "accepted"})
        setup_db.add(run)
        await setup_db.flush()
        gate = OrchestrationGate(
            run_id=run.id, success_criterion_key="plan_item:paused-restart", gate_type="work_completed",
            required_evidence={"required_source_types": ["task"], "min_count": 1},
            status="open",
        )
        setup_db.add(gate)
        await setup_db.flush()
        run_id, gate_id = run.id, gate.id

    async with sessions.begin() as repair_db:
        await OrchestrationService().tick(repair_db, run_id)
        async with sessions() as observer_db:
            gate = await observer_db.get(OrchestrationGate, gate_id)
            assert gate is not None
            assert gate.required_evidence["required_source_types"] == ["task"]

    async with sessions() as verify_db:
        gate = await verify_db.get(OrchestrationGate, gate_id)
        run = await verify_db.get(OrchestrationRun, run_id)
        repairs = list((await verify_db.scalars(select(EventLog).where(
            EventLog.event_type == "orchestration.gate_repaired"
        ))).all())
        assert gate is not None and run is not None
        assert gate.status == "open"
        assert gate.required_evidence["required_source_types"] == ["task", "verification"]
        assert run.status == "paused"
        assert len(repairs) == 0


async def test_outcome_sessionless_producer_and_verifier_accept_gate(
    db_session, test_project, stub_decision, safe_effectiveness_review_continue
):
    producer = _agent("sessionless-producer", ["implementation"])
    verifier = _agent("sessionless-verifier", ["validation"])
    producer.role, verifier.role = "developer", "validator"
    db_session.add_all([producer, verifier])
    await db_session.flush()
    service, _goal, run = await _authorized_run(db_session, test_project)
    await _seed_accepted_plan(db_session, test_project, service, run, producer, plan_items=[{
        "id": "sessionless-producer", "work_function": "implementation",
        "scope": "Deliver work without a session.", "deliverable": "Delivered work.", "agent_id": str(producer.id),
        "required_evidence": {"required_source_types": ["task", "review"], "min_count": 2},
    }])
    await service.tick(db_session, run.id)
    work_task = next(task for task in await _tasks_for_run(db_session, run.id)
                     if service._task_work_function(task) == "implementation")
    gate = await db_session.get(OrchestrationGate, uuid.UUID(work_task.metadata_["orchestration"]["plan_item_gate_id"]))
    assert gate.required_evidence["required_source_types"] == ["task", "verification"]
    assert gate.required_evidence["min_count"] == 2
    work_task.status = "done"
    await emit_event_once(
        db_session, test_project.id, "task.status_changed",
        {"task_id": str(work_task.id), "status": "done", "previous_status": "in_progress"},
        dedup_key=f"runtime-e2e-sessionless-producer:{work_task.id}",
    )
    action = await service.execute_request_verification_action(
        db_session, run.id,
        {"action_type": "request_verification", "gate_id": str(gate.id), "work_function": "validation"},
        f"run:{run.id}:kind:sessionless-producer-verification",
    )
    verification_task = await db_session.get(Task, action.target_id)
    assert verification_task is not None
    await _complete_task_session(
        db_session, test_project.id, verification_task, verifier.id,
        json.dumps({"status": "done", "verdict": "accepted", "evidence": ["Independent evidence."]}),
    )
    stub_decision(lambda _ctx: {"action_type": "noop"})
    await service.tick(db_session, run.id)
    await db_session.refresh(gate)
    assert gate.status == "accepted"


@pytest.mark.parametrize("failure", ["wrong_action", "wrong_gate", "wrong_source", "wrong_agent", "graph", "stale", "malformed"])
async def test_outcome_verification_rejects_unbound_or_invalid_sessions(
    db_session, test_project, stub_decision, safe_effectiveness_review_continue, failure
):
    producer = _agent(f"verification-producer-{failure}", ["implementation"])
    verifier = _agent(f"verification-verifier-{failure}", ["validation"])
    producer.role, verifier.role = "developer", "validator"
    db_session.add_all([producer, verifier])
    await db_session.flush()
    service, _goal, run = await _authorized_run(db_session, test_project)
    await _seed_accepted_plan(db_session, test_project, service, run, producer, plan_items=[{
        "id": f"verification-{failure}", "work_function": "implementation",
        "scope": "Deliver work.", "deliverable": "Delivered work.", "agent_id": str(producer.id),
    }])
    await service.tick(db_session, run.id)
    work_task = next(task for task in await _tasks_for_run(db_session, run.id)
                     if service._task_work_function(task) == "implementation")
    gate = await db_session.get(OrchestrationGate, uuid.UUID(work_task.metadata_["orchestration"]["plan_item_gate_id"]))
    await _complete_task_session(db_session, test_project.id, work_task, producer.id,
                                 json.dumps({"status": "done", "changes": ["Delivered work."]}))
    action = await service.execute_request_verification_action(
        db_session, run.id,
        {"action_type": "request_verification", "gate_id": str(gate.id), "work_function": "validation"},
        f"run:{run.id}:kind:invalid-verification:{failure}",
    )
    verification_task = await db_session.get(Task, action.target_id)
    assert verification_task is not None
    if failure == "wrong_action":
        action.action_type = "noop"
    elif failure == "wrong_gate":
        action.request = {**action.request, "gate_id": str(uuid.uuid4())}
    elif failure == "wrong_source":
        action.request = {**action.request, "source_task_id": str(uuid.uuid4())}
    elif failure == "graph":
        graph = Graph(project_id=test_project.id, name=f"bad-{uuid.uuid4()}", description="", definition={}, triggers=[])
        db_session.add(graph)
        await db_session.flush()
        instance = GraphRun(graph_id=graph.id, project_id=test_project.id,
                                    linked_task_id=verification_task.id, current_node="done", status="completed")
        db_session.add(instance)
        await db_session.flush()
        verification_task.graph_run_id = instance.id
    agent_id = producer.id if failure == "wrong_agent" else verifier.id
    output = "not-json" if failure == "malformed" else json.dumps({
        "status": "done", "verdict": "accepted", "evidence": ["Independent evidence."],
    })
    session = await _complete_task_session(
        db_session, test_project.id, verification_task, agent_id, output,
        graph_run_id=None,
    )
    if failure == "stale":
        db_session.add(Session(task_id=verification_task.id, agent_id=verifier.id, project_id=test_project.id,
                               adapter_type="api", status="completed", output="{}", metadata_={}, origin="auto"))
        await db_session.flush()
    stub_decision(lambda _ctx: {"action_type": "noop"})
    await service.tick(db_session, run.id)
    await db_session.refresh(gate)
    assert gate.status == "open"
    evidence = list((await db_session.scalars(
        select(OrchestrationEvidence).where(OrchestrationEvidence.gate_id == gate.id)
    )).all())
    assert not any(item.source_type == "verification" and item.source_id == session.id for item in evidence)


@pytest.mark.parametrize("output", [
    "{}",
    json.dumps({"status": "blocked", "verdict": "accepted", "evidence": ["evidence"]}),
    json.dumps({"status": "done", "verdict": "candidate", "evidence": ["evidence"]}),
    json.dumps({"status": "done", "verdict": "accepted", "evidence": []}),
    json.dumps({"status": "done", "verdict": "accepted", "evidence": ["  "]}),
    json.dumps({"status": "done", "verdict": "accepted", "evidence": [1]}),
])
async def test_outcome_verification_report_rejects_incomplete_structured_output(output):
    assert OrchestrationService._verification_report(output) is None


async def test_non_outcome_plan_item_gate_preserves_configured_evidence(db_session, test_project):
    goal = OrchestrationGoal(
        project_id=test_project.id,
        objective="Preserve roadmap evidence semantics.",
        goal_type="roadmap",
        success_criteria=[],
        constraints={},
        budget={},
    )
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id, plan_state={"status": "accepted"})
    db_session.add(run)
    await db_session.flush()
    configured = {"required_source_types": ["task", "review"], "min_count": 2}
    gate = OrchestrationGate(
        run_id=run.id,
        success_criterion_key="plan_item:roadmap",
        gate_type="work_completed",
        required_evidence=configured,
    )
    db_session.add(gate)
    await db_session.flush()

    assert await OrchestrationService()._required_evidence_for_gate(db_session, gate) == configured
    assert gate.required_evidence == configured


async def test_outcome_plan_item_gate_ignores_generic_same_agent_review(
    db_session, test_project, stub_decision, safe_effectiveness_review_continue
):
    """A spoofed generic review has no Outcome gate authority."""
    producer = _agent("same-agent-producer", ["implementation", "validation"])
    producer.role = "developer"
    db_session.add(producer)
    await db_session.flush()
    service, _goal, run = await _authorized_run(db_session, test_project)
    await _seed_accepted_plan(
        db_session,
        test_project,
        service,
        run,
        producer,
        plan_items=[{
            "id": "same-agent-gate",
            "work_function": "implementation",
            "scope": "Deliver work requiring independent verification.",
            "deliverable": "A reviewed result.",
            "agent_id": str(producer.id),
            "required_evidence": {"required_source_types": ["task"], "min_count": 1},
        }],
    )
    await service.tick(db_session, run.id)
    work_task = next(task for task in await _tasks_for_run(db_session, run.id)
                     if service._task_work_function(task) == "implementation")
    gate = await db_session.get(
        OrchestrationGate, uuid.UUID(work_task.metadata_["orchestration"]["plan_item_gate_id"])
    )
    await _complete_task_session(
        db_session, test_project.id, work_task, producer.id,
        json.dumps({"status": "done", "changes": ["Delivered work."]}),
    )
    await _complete_task_session(
        db_session, test_project.id, work_task, producer.id,
        json.dumps({"status": "done", "changes": ["Self-reviewed work."]}),
        emit_task_event=False,
        event_type="review.approved",
    )
    stub_decision(lambda _ctx: {"action_type": "noop"})
    await service.tick(db_session, run.id)
    await db_session.refresh(gate)
    assert gate.status == "open"


async def test_exhausted_budget_blocks_release_and_persists_attention_state(
    db_session, test_project, safe_effectiveness_review_continue
):
    """A budget stop must prevent new work, not merely block final closeout."""
    worker = _agent("budget-worker", ["implementation"])
    worker.role = "developer"
    db_session.add(worker)
    await db_session.flush()
    service, _goal, run = await _authorized_run(db_session, test_project)
    await _seed_accepted_plan(db_session, test_project, service, run, worker)
    run.budget_state = {"status": "exceeded", "overridden": False}
    await db_session.flush()
    actions = await _run_actions(db_session, run.id)

    tick = await service.tick(db_session, run.id)
    await db_session.refresh(run)

    assert tick["authorized_execution"] == {"step": "budget_exhausted"}
    assert all(
        service._task_work_function(task) == "planning"
        for task in await _tasks_for_run(db_session, run.id)
    )
    assert any(blocker["kind"] == "budget_exhausted" for blocker in run.active_blockers)
    warnings = list(await db_session.scalars(
        select(OrchestrationWarning).where(
            OrchestrationWarning.run_id == run.id,
            OrchestrationWarning.warning_type == "budget_exhausted",
        )
    ))

    assert len(warnings) == 1
    assert len(await _run_actions(db_session, run.id)) == len(actions)

    second_tick = await service.tick(db_session, run.id)
    warnings_after = list(await db_session.scalars(
        select(OrchestrationWarning).where(
            OrchestrationWarning.run_id == run.id,
            OrchestrationWarning.warning_type == "budget_exhausted",
        )
    ))

    assert second_tick["authorized_execution"] == {"step": "budget_exhausted"}
    assert len(await _run_actions(db_session, run.id)) == len(actions)
    assert len(warnings_after) == len(warnings)


async def test_budget_override_allows_authorized_work_release(
    db_session, test_project, safe_effectiveness_review_continue
):
    worker = _agent("budget-override-worker", ["implementation"])
    worker.role = "developer"
    db_session.add(worker)
    await db_session.flush()
    service, _goal, run = await _authorized_run(db_session, test_project)
    await _seed_accepted_plan(
        db_session,
        test_project,
        service,
        run,
        worker,
        plan_items=[{
            "id": "active-before-budget-stop",
            "work_function": "implementation",
            "scope": "Complete already released work.",
            "deliverable": "A terminal work report.",
            "agent_id": str(worker.id),
        }],
    )
    run.budget_state = {"status": "exceeded", "overridden": False}
    await db_session.flush()

    await service.tick(db_session, run.id)
    run.budget_state = {"status": "exceeded", "overridden": True}
    await db_session.flush()

    tick = await service.tick(db_session, run.id)
    warnings = list(await db_session.scalars(
        select(OrchestrationWarning).where(
            OrchestrationWarning.run_id == run.id,
            OrchestrationWarning.warning_type == "budget_exhausted",
        )
    ))

    assert tick["authorized_execution"]["step"] == "release_work"
    assert not any(blocker["kind"] == "budget_exhausted" for blocker in run.active_blockers)
    assert len(warnings) == 1 and not warnings[0].active


async def test_exhausted_budget_skips_recovery_work(
    db_session, test_project, safe_effectiveness_review_continue, monkeypatch
):
    worker = _agent("budget-recovery-worker", ["implementation"])
    worker.role = "developer"
    db_session.add(worker)
    await db_session.flush()
    service, _goal, run = await _authorized_run(db_session, test_project)
    await _seed_accepted_plan(
        db_session, test_project, service, run, worker,
        plan_items=[{
            "id": "recoverable-work",
            "work_function": "implementation",
            "scope": "Deliver recoverable work.",
            "deliverable": "A recoverable result.",
            "agent_id": str(worker.id),
        }],
    )
    await service.tick(db_session, run.id)
    work_task = next(
        task for task in await _tasks_for_run(db_session, run.id)
        if service._task_work_function(task) == "implementation"
    )
    work_task.status = "failed"
    run.budget_state = {"status": "exceeded", "overridden": False}
    await db_session.flush()
    actions = await _run_actions(db_session, run.id)

    async def unexpected_review(*_args, **_kwargs):
        raise AssertionError("exhausted budget must skip effectiveness review")

    monkeypatch.setattr(EffectivenessReviewProcess, "advance", unexpected_review)

    tick = await service.tick(db_session, run.id)

    assert tick["recoveries_created"] == 0
    assert work_task.status == "failed"
    assert len(await _run_actions(db_session, run.id)) == len(actions)


async def test_exhausted_budget_still_ingests_terminal_work_without_dispatch(
    db_session, test_project, safe_effectiveness_review_continue, monkeypatch
):
    worker = _agent("budget-active-worker", ["implementation"])
    worker.role = "developer"
    db_session.add(worker)
    await db_session.flush()
    service, _goal, run = await _authorized_run(db_session, test_project)
    await _seed_accepted_plan(
        db_session,
        test_project,
        service,
        run,
        worker,
        plan_items=[{
            "id": "active-before-budget-stop",
            "work_function": "implementation",
            "scope": "Complete already released work.",
            "deliverable": "A terminal work report.",
            "agent_id": str(worker.id),
        }],
    )
    release_tick = await service.tick(db_session, run.id)
    tasks = await _tasks_for_run(db_session, run.id)
    assert release_tick["authorized_execution"]["step"] == "release_work", release_tick
    task = next(task for task in tasks
                if service._task_work_function(task) == "implementation")
    gate = await db_session.get(
        OrchestrationGate, uuid.UUID(task.metadata_["orchestration"]["plan_item_gate_id"])
    )
    run.budget_state = {"status": "exceeded", "overridden": False}
    await _complete_task_session(
        db_session,
        test_project.id,
        task,
        worker.id,
        json.dumps({"status": "done", "changes": ["Finished before the budget stop."]}),
    )
    actions_before = await _run_actions(db_session, run.id)

    async def llm_must_not_run(*_args, **_kwargs):
        raise AssertionError("exhausted budget must not request an LLM decision")

    monkeypatch.setattr(service, "request_llm_decision", llm_must_not_run)
    tick = await service.tick(db_session, run.id)

    await db_session.refresh(run)
    await db_session.refresh(gate)
    assert tick["authorized_execution"] == {"step": "budget_exhausted"}
    assert tick["evidence_created"] == 2
    assert tick["gates_validated"] == 0
    assert tick["event_cursor"] is not None
    assert gate.status == "open"
    action_ids_before = {action.id for action in actions_before}
    new_actions = [action for action in await _run_actions(db_session, run.id)
                   if action.id not in action_ids_before]
    assert [action.action_type for action in new_actions] == ["report_consumed"]


async def test_malformed_report_curates_nothing_and_verifier_alone_accepts(
    db_session, test_project, stub_decision, safe_effectiveness_review_continue
):
    producer = _agent("malformed-producer", ["implementation"])
    verifier = _agent("malformed-verifier", ["validation"])
    producer.role = "developer"
    verifier.role = "validator"
    db_session.add_all([producer, verifier])
    await db_session.flush()
    service, goal, run = await _authorized_run(db_session, test_project)
    await _seed_accepted_plan(
        db_session,
        test_project,
        service,
        run,
        producer,
        plan_items=[{
            "id": "malformed-report",
            "work_function": "implementation",
            "scope": "Return a report that cannot be parsed.",
            "deliverable": "A verifier-reviewed result.",
            "agent_id": str(producer.id),
            "required_evidence": {
                "required_source_types": ["review"],
                "min_count": 1,
                "requires_independent_agent": True,
                "work_producer_agent_id": str(producer.id),
            },
        }],
    )
    await service.tick(db_session, run.id)
    task = next(task for task in await _tasks_for_run(db_session, run.id)
                if service._task_work_function(task) == "implementation")
    gate = await db_session.get(
        OrchestrationGate, uuid.UUID(task.metadata_["orchestration"]["plan_item_gate_id"])
    )
    await _complete_task_session(db_session, test_project.id, task, producer.id, "{not json")
    stub_decision(lambda _ctx: {"action_type": "noop"})
    await service.tick(db_session, run.id)

    sections = await OrchestrationMemoryService().list_sections(
        db_session, test_project.id, goal.id
    )
    assert gate.status == "open"
    assert not [section for section in sections if section.section_key.startswith("report_")]
    assert any(action.action_type == "report_consumed" for action in await _run_actions(db_session, run.id))

    request = await service.execute_request_verification_action(
        db_session,
        run.id,
        {"action_type": "request_verification", "gate_id": str(gate.id), "work_function": "validation"},
        f"run:{run.id}:kind:malformed-report-verification",
    )
    verification_task = await db_session.get(Task, request.target_id)
    assert verification_task is not None and verification_task.assigned_to == verifier.id
    await _complete_task_session(
        db_session,
        test_project.id,
        verification_task,
        verifier.id,
        json.dumps({"status": "done", "verdict": "accepted", "evidence": ["Verifier independently accepted the work."]}),
    )
    await service.tick(db_session, run.id)

    await db_session.refresh(gate)
    assert gate.status == "accepted"


async def test_pause_retains_active_work_and_resume_reconciles_it(
    db_session, test_project, stub_decision, safe_effectiveness_review_continue
):
    worker = _agent("pause-worker", ["implementation"])
    worker.role = "developer"
    db_session.add(worker)
    await db_session.flush()
    service, goal, run = await _authorized_run(db_session, test_project)
    await _seed_accepted_plan(
        db_session,
        test_project,
        service,
        run,
        worker,
        plan_items=[{
            "id": "pause-work",
            "work_function": "implementation",
            "scope": "Complete work across a pause.",
            "deliverable": "A reconciled terminal result.",
            "agent_id": str(worker.id),
        }],
    )
    await service.tick(db_session, run.id)
    task = next(task for task in await _tasks_for_run(db_session, run.id)
                if service._task_work_function(task) == "implementation")

    await service.pause_goal(db_session, test_project.id, goal.id)
    paused_tick = await service.tick(db_session, run.id)
    assert paused_tick["processed_events"] == 0
    assert task.status in {"backlog", "ready", "in_progress"}

    await service.resume_goal(db_session, test_project.id, goal.id)
    await _complete_task_session(
        db_session,
        test_project.id,
        task,
        worker.id,
        json.dumps({"status": "done", "changes": ["Completed after resume."]}),
    )
    stub_decision(lambda _ctx: {"action_type": "noop"})
    resumed_tick = await service.tick(db_session, run.id)
    gate = await db_session.get(
        OrchestrationGate, uuid.UUID(task.metadata_["orchestration"]["plan_item_gate_id"])
    )
    assert resumed_tick["evidence_created"] == 2
    assert gate.status == "open"


async def test_runtime_recovery_retries_reassigns_then_escalates_needs_attention(
    db_session, test_engine, test_project, stub_decision, safe_effectiveness_review_continue
):
    original = _agent("recovery-original", ["implementation"])
    alternate = _agent("recovery-alternate", ["implementation"])
    original.role = alternate.role = "developer"
    db_session.add_all([original, alternate])
    await db_session.flush()
    service, goal, run = await _authorized_run(db_session, test_project)
    await _seed_accepted_plan(
        db_session,
        test_project,
        service,
        run,
        original,
        plan_items=[{
            "id": "recovery-work",
            "work_function": "implementation",
            "scope": "Exercise bounded runtime recovery.",
            "deliverable": "A recoverable task.",
            "agent_id": str(original.id),
        }],
    )
    stub_decision(lambda _ctx: {"action_type": "noop"})
    await service.tick(db_session, run.id)
    task = next(task for task in await _tasks_for_run(db_session, run.id)
                if service._task_work_function(task) == "implementation")

    async def fail_attempt(agent_id):
        task.status = "failed"
        db_session.add(Session(
            project_id=test_project.id,
            task_id=task.id,
            agent_id=agent_id,
            adapter_type="api",
            status="failed",
            error="transient failure",
            metadata_={},
            origin="auto",
        ))
        await db_session.flush()

    await fail_attempt(original.id)
    await service.tick(db_session, run.id)
    retry = next(action for action in await _run_actions(db_session, run.id)
                 if action.action_type == "retry_task")
    assert retry.status == "completed" and task.status == "in_progress"

    retried_session = await db_session.get(Session, retry.target_id)
    task.status = retried_session.status = "failed"
    await db_session.flush()
    await service.tick(db_session, run.id)
    reassign = next(action for action in await _run_actions(db_session, run.id)
                    if action.action_type == "reassign_task")
    assert reassign.status == "completed" and task.assigned_to == alternate.id, reassign.error

    reassigned_session = await db_session.get(Session, reassign.target_id)
    task.status = reassigned_session.status = "failed"
    await db_session.flush()
    await service.tick(db_session, run.id)
    assert goal.status == run.status == "blocked"
    blocker = next(item for item in run.active_blockers if item["kind"] == "repeated_failure")
    assert {key: value for key, value in blocker.items() if key != "decision_id"} == {
        "kind": "repeated_failure",
        "task_id": str(task.id),
        "gate_id": task.metadata_["orchestration"]["plan_item_gate_id"],
        "attempt_count": 3,
        "failed_session_ids": [
            str(session.id)
            for session in (await db_session.scalars(
                select(Session).where(Session.task_id == task.id, Session.status == "failed").order_by(Session.id)
            )).all()
        ],
        "owner": "human",
        "reason": "Task failed repeatedly after retry and reassignment attempts.",
        "recommended_action": "Review the failed attempts and choose how to proceed.",
    }
    asks = [action for action in await _run_actions(db_session, run.id) if action.action_type == "ask_human"]
    assert len(asks) == 1
    assert asks[0].request["reason"] == blocker["reason"]
    assert not any(action.action_type == "pause_run" for action in await _run_actions(db_session, run.id))
    assert service.run_condition(goal, run) == "needs_attention"

    action_count = len(await _run_actions(db_session, run.id))
    event_count = len((await db_session.scalars(select(EventLog).where(
        EventLog.project_id == test_project.id,
        EventLog.event_type == "orchestration.human_input_required",
    ))).all())
    await db_session.commit()
    sessions = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with sessions.begin() as restart_db:
        replay = await OrchestrationService().tick(restart_db, run.id)
        replay_run = await restart_db.get(OrchestrationRun, run.id)
        replay_goal = await restart_db.get(OrchestrationGoal, goal.id)
        replay_actions = await _run_actions(restart_db, run.id)
        assert replay.get("authorized_execution") is None
        assert replay_run is not None and replay_goal is not None
        assert replay_run.status == replay_goal.status == "blocked"
        assert OrchestrationService().run_condition(replay_goal, replay_run) == "needs_attention"
        assert len(replay_actions) == action_count
        assert len([action for action in replay_actions if action.action_type == "ask_human"]) == 1
        assert len((await restart_db.scalars(select(EventLog).where(
            EventLog.project_id == test_project.id,
            EventLog.event_type == "orchestration.human_input_required",
        ))).all()) == event_count


@pytest.mark.parametrize("run_status", ["running", "blocked"])
async def test_reconciliation_repairs_weak_accepted_outcome_gate_once_then_bound_verification_reaccepts(
    db_session, test_project, stub_decision, safe_effectiveness_review_continue, run_status
):
    """Removing the repair must leave legacy acceptance trusted and fail this test."""
    producer = _agent("repair-producer", ["implementation"])
    verifier = _agent("repair-verifier", ["validation"])
    producer.role, verifier.role = "developer", "validator"
    db_session.add_all([producer, verifier])
    await db_session.flush()
    service, goal, run = await _authorized_run(db_session, test_project)
    await _seed_accepted_plan(
        db_session,
        test_project,
        service,
        run,
        producer,
        plan_items=[{
            "id": "legacy-weak-gate",
            "work_function": "implementation",
            "scope": "Repair legacy gate acceptance.",
            "deliverable": "Bound verification.",
            "agent_id": str(producer.id),
        }],
    )
    stub_decision(lambda _ctx: {"action_type": "noop"})
    await service.tick(db_session, run.id)
    if run_status == "blocked":
        goal.status = run.status = "blocked"
        await db_session.flush()
    await db_session.refresh(run)
    assert run.status == run_status
    work_task = next(task for task in await _tasks_for_run(db_session, run.id)
                     if service._task_work_function(task) == "implementation")
    gate = await db_session.get(
        OrchestrationGate, uuid.UUID(work_task.metadata_["orchestration"]["plan_item_gate_id"])
    )
    weak = OrchestrationEvidence(
        run_id=run.id,
        gate_id=gate.id,
        source_type="task",
        source_id=uuid.uuid4(),
        producer_agent_id=producer.id,
        verdict="accepted",
        evidence_metadata={},
    )
    gate.status, gate.accepted_at = "accepted", _utcnow()
    db_session.add(weak)
    await db_session.flush()

    # Reconciliation reopens the weak acceptance before ordinary gate validation.
    stub_decision(lambda _ctx: {"action_type": "noop"})
    await OrchestrationService().tick(db_session, run.id)
    await db_session.refresh(gate)
    await db_session.refresh(weak)
    repairs = list((await db_session.scalars(select(EventLog).where(
        EventLog.event_type == "orchestration.gate_repaired"
    ))).all())
    assert gate.status == "open"
    assert gate.accepted_at is None
    assert weak.verdict == "candidate"
    assert weak.evidence_metadata["repair_reason"] == "missing_bound_verification"
    assert len(repairs) == 1

    # A legacy verification row has the right source type but no bound action/session.
    unbound = OrchestrationEvidence(
        run_id=run.id,
        gate_id=gate.id,
        source_type="verification",
        source_id=uuid.uuid4(),
        producer_agent_id=verifier.id,
        verdict="candidate",
        evidence_metadata={},
        created_at=weak.created_at,
    )
    db_session.add(unbound)
    await db_session.flush()
    await service.validate_open_gates(db_session, run.id)
    await db_session.refresh(gate)
    assert gate.status == "open"
    await db_session.delete(unbound)
    await db_session.flush()

    # Replay stays quiet, then only a real bound verifier can re-accept the gate.
    stub_decision(lambda _ctx: {"action_type": "noop"})
    await OrchestrationService().tick(db_session, run.id)
    assert len(list((await db_session.scalars(select(EventLog).where(
        EventLog.event_type == "orchestration.gate_repaired"
    ))).all())) == 1

    await _complete_task_session(
        db_session, test_project.id, work_task, producer.id,
        json.dumps({"status": "done", "changes": ["Delivered work."]}),
    )
    action = await service.execute_request_verification_action(
        db_session,
        run.id,
        {"action_type": "request_verification", "gate_id": str(gate.id), "work_function": "validation"},
        f"run:{run.id}:kind:repair-verification",
    )
    verification_task = await db_session.get(Task, action.target_id)
    assert verification_task is not None
    await _complete_task_session(
        db_session, test_project.id, verification_task, verifier.id,
        json.dumps({"status": "done", "verdict": "accepted", "evidence": ["Independent proof."]}),
    )
    stub_decision(lambda _ctx: {"action_type": "noop"})
    await service.tick(db_session, run.id)
    await db_session.refresh(gate)
    assert gate.status == "accepted"
    await OrchestrationService().tick(db_session, run.id)
    await db_session.refresh(gate)
    assert gate.status == "accepted"
    assert len(list((await db_session.scalars(select(EventLog).where(
        EventLog.event_type == "orchestration.gate_repaired"
    ))).all())) == 1


@pytest.mark.parametrize(
    ("report", "case"),
    [
        (json.dumps({"status": "done"}), "structurally_incomplete"),
        (
            json.dumps({"status": "done", "criterion_progress": {"done": "claimed"}}),
            "claims_only",
        ),
    ],
)
async def test_incomplete_or_claim_only_report_is_consumed_once_without_curation_or_gate_acceptance(
    db_session,
    test_project,
    stub_decision,
    safe_effectiveness_review_continue,
    report,
    case,
):
    parsed = parse_work_report(report, {})
    assert parsed is not None
    assert parsed.status == "done"
    if case == "claims_only":
        assert parsed.criterion_progress == {"done": "claimed"}
    else:
        assert parsed.criterion_progress == {}

    producer = _agent(f"report-{case}", ["implementation"])
    producer.role = "developer"
    db_session.add(producer)
    await db_session.flush()
    service, goal, run = await _authorized_run(db_session, test_project)
    await _seed_accepted_plan(
        db_session,
        test_project,
        service,
        run,
        producer,
        plan_items=[{
            "id": f"report-{case}",
            "work_function": "implementation",
            "scope": "Submit a report with no curatable work facts.",
            "deliverable": "Independent verification remains required.",
            "agent_id": str(producer.id),
        }],
    )
    await service.tick(db_session, run.id)
    task = next(task for task in await _tasks_for_run(db_session, run.id)
                if service._task_work_function(task) == "implementation")
    gate = await db_session.get(
        OrchestrationGate, uuid.UUID(task.metadata_["orchestration"]["plan_item_gate_id"])
    )
    session = await _complete_task_session(
        db_session, test_project.id, task, producer.id, report
    )
    stub_decision(lambda _ctx: {"action_type": "noop"})
    first = await service.tick(db_session, run.id)

    completed_event = next(
        event
        for event in await db_session.scalars(
            select(EventLog).where(
                EventLog.project_id == test_project.id,
                EventLog.event_type == "session.completed",
            )
        )
        if event.payload.get("session_id") == str(session.id)
    )
    run.event_cursor = completed_event.seq - 1
    await db_session.flush()
    second = await service.tick(db_session, run.id)

    markers = [
        action for action in await _run_actions(db_session, run.id)
        if action.action_type == "report_consumed" and action.request.get("task_id") == str(task.id)
    ]
    sections = await OrchestrationMemoryService().list_sections(
        db_session, test_project.id, goal.id
    )
    await db_session.refresh(gate)
    assert first["evidence_created"] == 2
    assert second["evidence_created"] == 0
    assert len(markers) == 1
    assert not [section for section in sections if section.section_key == f"report_{session.id}"]
    assert gate.status == "open"
