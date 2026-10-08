"""End-to-end tests for the proactive orchestrator (spec 7.9/7.10): act-until-wait,
wake_when waits, the idle/due paths and the backstop. Drives tick() on the seeded
db_session and stubs only the LLM (the stub_decision fixture, or the adapter's
completion function for the real-adapter test)."""
import uuid
from datetime import timedelta

import pytest
from sqlalchemy import select

from huddleroom.config import settings
from huddleroom.models.base import _utcnow
from huddleroom.models.orchestration import (
    OrchestrationAction,
    OrchestrationDecision,
    OrchestrationGoal,
    OrchestrationRun,
    OrchestrationWait,
)
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.orchestration_supervision import SupervisionAssessment
from huddleroom.services.orchestration_wake_when import (
    BACKSTOP_RECHECK_SECONDS,
    ORCHESTRATOR_WAIT_OWNER_TYPE,
    WAKE_RECHECK_EVENT_TYPE,
    clamp_recheck_seconds,
)

from tests.conftest import heal_baseline_drift_for_test
from tests.orchestration_wake_helpers import NOOP_WAKE_WHEN
from tests.test_orchestration_roadmap_e2e import _ready_roadmap
from tests.test_orchestration_roadmap_task_items import roadmap_task
from tests.test_orchestration_runtime_e2e import (
    _agent,
    _authorized_run,
    _completed_parent_task,
    _planner_agent_id,
    _run_actions,
    _seed_accepted_plan,
    _tasks_for_run,
)

pytestmark = pytest.mark.asyncio

PLAN_WAKE = {"recheck_after_seconds": 600, "expected_result": "Plan task finishes"}


def _request_plan(ctx):
    return {
        "action_type": "request_plan",
        "agent_id": _planner_agent_id(ctx),
        "scope": "Produce a plan for the objective",
        "work_function": "planning",
        "reason": "No plan yet",
    }


def _plan_then_wait_stub(calls):
    """Call 1 request_plan, every later call a noop that names its wake."""
    def decide(ctx):
        calls.append(ctx)
        if len(calls) == 1:
            return _request_plan(ctx)
        return {"action_type": "noop", "reason": "Plan task in flight", "wake_when": dict(PLAN_WAKE)}
    return decide


async def _open_waits(db, run, owner_type=None):
    rows = (await db.scalars(
        select(OrchestrationWait).where(
            OrchestrationWait.run_id == run.id, OrchestrationWait.status == "open"
        ).order_by(OrchestrationWait.created_at, OrchestrationWait.wait_key)
    )).all()
    if owner_type is None:
        return list(rows)
    return [w for w in rows if (w.owner or {}).get("type") == owner_type]


async def _planned_and_waiting(db_session, test_project, stub_decision):
    """Authorized run after one tick that requested a plan and named its wake."""
    planner = _agent("planner", ["planning"])
    db_session.add(planner)
    await db_session.flush()
    service, goal, run = await _authorized_run(db_session, test_project)
    calls: list = []
    stub_decision(_plan_then_wait_stub(calls))
    result = await service.tick(db_session, run.id)
    await db_session.refresh(run)
    return service, goal, run, calls, result


async def test_fresh_authorized_run_requests_plan_then_names_its_wake(
    db_session, test_project, stub_decision
):
    service, goal, run, calls, result = await _planned_and_waiting(
        db_session, test_project, stub_decision
    )

    execution = result["authorized_execution"]
    assert len(execution["action_ids"]) == 2
    assert len(set(execution["action_ids"])) == 2
    assert run.plan_state["status"] == "requested"
    assert len(calls) == 2
    waits = await _open_waits(db_session, run)
    assert len(waits) == 1
    assert waits[0].owner["type"] == ORCHESTRATOR_WAIT_OWNER_TYPE
    assert waits[0].awaited_event["event_type"] == WAKE_RECHECK_EVENT_TYPE


async def test_idle_tick_after_wait_makes_no_adapter_call(db_session, test_project, stub_decision):
    service, goal, run, calls, _ = await _planned_and_waiting(db_session, test_project, stub_decision)
    before = len(calls)

    result = await service.tick(db_session, run.id)

    execution = result["authorized_execution"]
    assert execution["step"] == "local_liveness"
    assert execution["outcome"] == "waiting"
    assert len(calls) == before


async def test_due_orchestrator_wait_wakes_the_orchestrator(db_session, test_project, stub_decision):
    service, goal, run, calls, _ = await _planned_and_waiting(db_session, test_project, stub_decision)
    wait = (await _open_waits(db_session, run))[0]
    wait.due_recheck_at = _utcnow() - timedelta(seconds=5)
    await db_session.flush()
    before = len(calls)

    due = await service.tick(db_session, run.id)

    assert due["authorized_execution"]["step"] == "local_liveness"
    assert due["authorized_execution"]["outcome"] == "due_fallback"
    assert len(calls) == before
    assert await _open_waits(db_session, run, ORCHESTRATOR_WAIT_OWNER_TYPE) == []

    await service.tick(db_session, run.id)

    assert len(calls) > before


async def test_noop_without_wake_when_is_rejected_then_backstop_wait(
    db_session, test_project, stub_decision
):
    service, goal, run = await _authorized_run(db_session, test_project)
    stub_decision(lambda ctx: {"action_type": "noop", "reason": "Nothing to do"})

    result = await service.tick(db_session, run.id)

    decision = (await db_session.scalars(
        select(OrchestrationDecision).where(OrchestrationDecision.run_id == run.id)
    )).all()[-1]
    assert decision.validator_status == "rejected"
    noops = [a for a in await _run_actions(db_session, run.id) if a.action_type == "noop"]
    assert noops, "fallback noop should be recorded"
    assert result["authorized_execution"]["action_id"]
    waits = await _open_waits(db_session, run, ORCHESTRATOR_WAIT_OWNER_TYPE)
    assert len(waits) == 1
    assert waits[0].wait_key.endswith(":recheck")
    assert waits[0].fallback["recheck_seconds"] == clamp_recheck_seconds(BACKSTOP_RECHECK_SECONDS)


async def test_real_adapter_exhausts_repairs_then_backstop(db_session, test_project, monkeypatch):
    import json

    calls = []

    async def always_bare_noop(**request):
        calls.append(request)
        content = json.dumps({"decision": {"action_type": "noop", "reason": "waiting"}})
        return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr(
        "huddleroom.services.orchestration_llm_decision_adapter.get_orchestration_completion",
        lambda completion_fn=None: completion_fn or always_bare_noop,
    )
    service, goal, run = await _authorized_run(db_session, test_project)

    await service.tick(db_session, run.id)

    assert len(calls) == 3
    decision = (await db_session.scalars(
        select(OrchestrationDecision).where(OrchestrationDecision.run_id == run.id)
    )).all()[-1]
    assert decision.parsed_decision["action_type"] == "invalid_llm_output"
    assert decision.validator_status == "rejected"
    waits = await _open_waits(db_session, run, ORCHESTRATOR_WAIT_OWNER_TYPE)
    assert len(waits) == 1
    assert waits[0].wait_key.endswith(":recheck")


async def test_decision_context_reaches_adapter_with_progress_view(
    db_session, test_project, stub_decision
):
    service, goal, run = await _authorized_run(
        db_session,
        test_project,
        success_criteria=[
            {"key": "alpha", "description": "Alpha is delivered."},
            {"key": "beta", "description": "Beta is delivered."},
        ],
    )
    seen: list = []

    def decide(ctx):
        seen.append(ctx)
        return {"action_type": "noop", "reason": "Look only", "wake_when": dict(NOOP_WAKE_WHEN)}

    stub_decision(decide)

    await service.tick(db_session, run.id)

    ctx = seen[0]
    assert {entry["criterion_key"] for entry in ctx["progress_view"]} == {"alpha", "beta"}
    assert ctx["untracked_follow_ups"] == []
    assert ctx["untracked_follow_ups_total"] == 0


async def test_unchanged_situation_reuses_decision(test_engine, tmp_path, stub_decision, monkeypatch):
    # Reuse only exists on the committed path (isolated read sessions must see the run), which the
    # rolled-back db_session fixture cannot reach -- so this test owns a committing session.
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    import huddleroom.services.orchestration_service as orchestration_service_module
    from huddleroom.models.project import Project

    (tmp_path / "ws").mkdir()
    session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(orchestration_service_module, "AsyncSessionLocal", session_factory)
    async with session_factory() as db:
        project = Project(name="Reuse", description="d", workspace_path=str((tmp_path / "ws").resolve()), config={})
        db.add(project)
        await db.flush()
        db.add(_agent("planner", ["planning"]))
        await db.flush()
        service, goal, run = await _authorized_run(db, project)
        await db.commit()
        calls: list = []

        def decide(ctx):
            calls.append(ctx)
            return {"action_type": "noop", "reason": "Wait", "wake_when": dict(NOOP_WAKE_WHEN)}

        stub_decision(decide)

        first = await service.request_llm_decision(db, run.id)
        second = await service.request_llm_decision(db, run.id)

        assert first.validator_status == "accepted"
        assert second.id == first.id
        assert len(calls) == 1, "unchanged context must reuse the accepted decision (spec 3.6)"


async def _bare_run(db, project):
    goal = OrchestrationGoal(project_id=project.id, objective="Keep moving", status="active", goal_type="outcome")
    db.add(goal)
    await db.flush()
    run = OrchestrationRun(goal_id=goal.id, status="running", phase="authorized")
    db.add(run)
    await db.flush()
    return goal, run


async def test_supervision_continue_waits_then_due_wakes(db_session, test_project):
    goal, run = await _bare_run(db_session, test_project)
    supervision = OrchestrationService().supervision
    task_id = str(uuid.uuid4())
    wake_when = {
        "events": [{"event_type": "task.status_changed", "matcher": {"task_id": task_id}}],
        "recheck_after_seconds": 600,
        "expected_result": "The task changes state",
    }
    await supervision.apply_disposition(
        db_session, goal, run,
        SupervisionAssessment(disposition={
            "action_type": "continue", "origin": "test", "reason": "Why", "expected_result": "Expected",
            "contract_version": "start", "request": {"wake_when": wake_when},
        }),
    )
    group = await _open_waits(db_session, run, ORCHESTRATOR_WAIT_OWNER_TYPE)
    assert len(group) == 2

    first = await supervision.reconcile_local(db_session, goal, run)
    assert first["outcome"] == "waiting"

    recheck = next(w for w in group if w.awaited_event["event_type"] == WAKE_RECHECK_EVENT_TYPE)
    recheck.due_recheck_at = _utcnow() - timedelta(seconds=1)
    await db_session.flush()

    due = await supervision.reconcile_local(db_session, goal, run)

    assert due["outcome"] == "due_fallback"
    assert {w.status for w in group} == {"cleared"}
    assert await _open_waits(db_session, run, ORCHESTRATOR_WAIT_OWNER_TYPE) == []
    assert (await supervision.reconcile_local(db_session, goal, run))["outcome"] != "waiting"


async def test_generation_suffix_after_clear_on_decision_path(db_session, test_project, stub_decision):
    service, goal, run = await _authorized_run(db_session, test_project)
    task_id = str(uuid.uuid4())
    stub_decision(lambda ctx: {
        "action_type": "noop",
        "reason": "Wait for the task",
        "wake_when": {
            "events": [{"event_type": "task.status_changed", "matcher": {"task_id": task_id}}],
            "expected_result": "The task changes state",
        },
    })
    await service.tick(db_session, run.id)
    decision = (await db_session.scalars(
        select(OrchestrationDecision).where(OrchestrationDecision.run_id == run.id)
        .order_by(OrchestrationDecision.created_at)
    )).all()[-1]
    assert decision.validator_status == "accepted"
    first = await _open_waits(db_session, run, ORCHESTRATOR_WAIT_OWNER_TYPE)
    assert len(first) == 1 and not first[0].wait_key.endswith(":g1")

    cleared = await service.supervision.clear_matching_waits(
        db_session, run, event_type="task.status_changed", subject_id=None,
        matcher={"task_id": task_id},
    )
    assert cleared == 1
    assert await _open_waits(db_session, run, ORCHESTRATOR_WAIT_OWNER_TYPE) == []

    await db_session.refresh(run)
    await service._dispatch_execution_decision(db_session, run, decision)

    reopened = await _open_waits(db_session, run, ORCHESTRATOR_WAIT_OWNER_TYPE)
    assert len(reopened) == 1
    assert reopened[0].wait_key.endswith(":g1")
    assert reopened[0].owner["id"] == str(decision.id)

    # A further replay with the wait still open must not duplicate it.
    await service._dispatch_execution_decision(db_session, run, decision)
    assert len(await _open_waits(db_session, run, ORCHESTRATOR_WAIT_OWNER_TYPE)) == 1


async def test_in_tick_follow_up_decisions_deduplicate_with_loop(
    db_session, test_project, stub_decision, monkeypatch
):
    monkeypatch.setattr(settings, "orchestration_max_actions_per_tick", 3)
    service, run, parent, agent, report_session = await _completed_parent_task(db_session, test_project)
    await _seed_accepted_plan(
        db_session, test_project, service, run, agent,
        plan_items=[{
            "id": "already-released-item",
            "work_function": "implementation",
            "scope": "A previously released item.",
            "deliverable": "No duplicate work should release.",
            "agent_id": str(agent.id),
        }],
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

    result = await service.tick(db_session, run.id)

    execution = result["authorized_execution"]
    action_ids = execution.get("action_ids") or [execution["action_id"]]
    assert len(action_ids) == len(set(action_ids))
    assert decision_calls >= 2, "the loop should have regenerated the follow-up in the same tick"
    followup_actions = [
        a for a in await _run_actions(db_session, run.id)
        if a.action_type == "create_delegation_task" and a.request.get("work_function") == "follow_up"
    ]
    followups = [t for t in await _tasks_for_run(db_session, run.id)
                 if service._task_work_function(t) == "follow_up"]
    assert len(followup_actions) == 1
    assert len(followups) == 1
    assert len(action_ids) == 1


async def test_nested_roadmap_outcome_loop_real(db_session, test_project, stub_decision, monkeypatch):
    from huddleroom.models.artifact import Artifact
    from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
    from huddleroom.services.orchestration_roadmap_service import OrchestrationRoadmapService

    cap = 3
    monkeypatch.setattr(settings, "orchestration_max_actions_per_tick", cap)
    service, goal, run, planner = await _ready_roadmap(db_session, test_project)
    assert await OrchestrationRoadmapService(service).current_version(db_session, goal.id) is None
    calls: list = []
    mode = {"accept_artifact": None}

    def decide(ctx):
        calls.append(ctx)
        if mode["accept_artifact"]:
            return {"action_type": "accept_plan", "plan_artifact_id": mode["accept_artifact"],
                    "reason": "Plan is ready"}
        if len(calls) == 1:
            return {
                "action_type": "request_plan", "agent_id": str(planner.id), "work_function": "planning",
                "scope": "Plan the roadmap", "reason": "No plan yet",
            }
        return {"action_type": "noop", "reason": "Plan task in flight", "wake_when": dict(PLAN_WAKE)}

    stub_decision(decide)

    # Phase 1: the roadmap `advance` wrapper must not iterate on the inner plan_decision result.
    inner_runs = []
    original = service._advance_authorized_execution

    async def spy(db, g, r):
        result = await original(db, g, r)
        inner_runs.append(result)
        return result

    monkeypatch.setattr(service, "_advance_authorized_execution", spy)
    result = await service.tick(db_session, run.id)

    execution = result["authorized_execution"]
    assert len(inner_runs) == 1, "outer roadmap loop must not re-enter the inner loop"
    assert execution == inner_runs[0]
    assert execution["step"] == "plan_decision"
    assert 1 < len(execution["action_ids"]) <= cap
    assert len(set(execution["action_ids"])) == len(execution["action_ids"])
    assert len(calls) == len(execution["action_ids"])
    await db_session.refresh(run)
    assert run.plan_state["status"] == "requested"
    waits = await _open_waits(db_session, run, ORCHESTRATOR_WAIT_OWNER_TYPE)
    assert len(waits) == 1

    # Phase 2: the plan is accepted through the roadmap approval path; the loop stops after accept_plan.
    plan_action = next(a for a in await _run_actions(db_session, run.id) if a.action_type == "request_plan")
    artifact = Artifact(
        project_id=test_project.id, name="e2e-roadmap-plan", artifact_type="plan", status="draft",
        linked_task_id=plan_action.target_id, created_by_agent=planner.id,
        metadata_={"plan_items": [{**roadmap_task("build"), "agent_id": str(planner.id)}]},
    )
    db_session.add(artifact)
    await db_session.flush()
    for wait in waits:
        wait.status = "cleared"
    mode["accept_artifact"] = str(artifact.id)
    await db_session.flush()

    from huddleroom.models.agent import Agent

    goal.orchestrator_context = {"team": {"agent_ids": [
        str(agent_id) for agent_id in await db_session.scalars(select(Agent.id).where(Agent.is_active.is_(True)))
    ]}}
    goal.weight = "trivial"  # keep the unrelated effectiveness analyzer deterministic
    await heal_baseline_drift_for_test(db_session, goal, run)
    pending_tick = await service.tick(db_session, run.id)
    assert pending_tick["authorized_execution"]["step"] == "waiting_plan_authority"
    decision = await db_session.scalar(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.goal_id == goal.id, OrchestrationAuthorityDecision.status == "pending"))
    assert decision is not None
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, decision, selected_option="approve", decided_by_user_id=goal.manager_user_id)
    before = len(calls)

    accepted_tick = await service.tick(db_session, run.id)

    await db_session.refresh(run)
    execution = accepted_tick["authorized_execution"]
    assert run.plan_state["status"] == "accepted", run.plan_state
    assert len(calls) == before + 1, "the loop must stop right after accept_plan"
    assert len(execution.get("action_ids") or [execution["action_id"]]) == 1
    assert not service._has_active_blocker(run, "plan_criterion_integrity")
    assert await OrchestrationRoadmapService(service).current_version(db_session, goal.id) is not None
