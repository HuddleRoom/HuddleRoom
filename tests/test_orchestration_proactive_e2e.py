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
from tests.test_orchestration_basic_recovery import (  # pylint: disable=unused-import
    _actions as _recovery_actions,
    _agent as _recovery_agent,
    _blocked_setup,
    _fake_task_run,
    _make_gate,
    _make_orchestrated_task,
    _make_run,
    _task_waits,
)
from tests.test_orchestration_decide_while_work import (  # pylint: disable=unused-import
    STUCK,
    _asks,
    _idle_setup,
    _situation,
    _ticks,
)
from tests.test_orchestration_goal_closeout import completion_ready_goal  # noqa: F401  pylint: disable=unused-import
from tests.test_orchestration_supervision_regressions import (
    _NOW,
    _fake_clock,
    _fresh_sessions,
    _live_claim_state,
    _persisted_state,
    _provider_assessment,
    _run as _bare_supervised_run,
)
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


# ---------------------------------------------------------------------------
# Deterministic end-to-end liveness scenarios.  Every scenario drives real
# services, stubs only LLM decisions / the semantic judge, asserts the global
# liveness invariant after each tick and replays the effects to prove idempotence.
# ---------------------------------------------------------------------------
_NON_SUBSTANTIVE_ACTIONS = {"noop", "record_warning", "ask_human", "decision_continuation"}


async def _snapshot(db, run):
    """Ids of every action and task of the run, taken before a tick."""
    return {a.id for a in await _run_actions(db, run.id)} | {t.id for t in await _tasks_for_run(db, run.id)}


async def _liveness_clause(db, goal, run, before):
    """Which arm of the invariant holds (substantive action/released work, bounded wait, human blocker), else None."""
    from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision

    await db.refresh(goal)
    await db.refresh(run)
    actions = await _run_actions(db, run.id)
    if any(a.id not in before and a.status == "completed" and a.action_type not in _NON_SUBSTANTIVE_ACTIONS
           for a in actions):
        return "action"
    if any(t.id not in before for t in await _tasks_for_run(db, run.id)):
        return "action"
    pending = (await db.scalars(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.run_id == run.id, OrchestrationAuthorityDecision.status == "pending",
    ))).all()
    if any((d.question or "").strip() and d.options for d in pending):
        return "human"
    if run.status == "running":
        for wait in await _open_waits(db, run):
            fallback = wait.fallback or {}
            owner = wait.owner or {}
            if (
                owner.get("type") and owner.get("id") and (wait.awaited_event or {}).get("event_type")
                and wait.due_recheck_at is not None
                and fallback.get("action_type") in {"continue", "attention"}
                and str(fallback.get("expected_result") or "").strip()
            ):
                return "wait"
    return None


async def _assert_liveness(db, goal, run, before, expect=None):
    clause = await _liveness_clause(db, goal, run, before)
    assert clause is not None, (
        "liveness invariant violated: no substantive action, bounded wait or actionable human blocker",
        run.status, [a.action_type for a in await _run_actions(db, run.id)],
    )
    if expect is not None:
        assert clause == expect
    return clause


async def _effect_counts(db, run):
    """Counts of every externally visible effect, for idempotent-replay comparison."""
    from huddleroom.models.graph import GraphRun
    from huddleroom.models.meeting import Meeting
    from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision

    actions = await _run_actions(db, run.id)
    return {
        "actions": len(actions),
        "action_types": sorted(a.action_type for a in actions),
        "tasks": len(await _tasks_for_run(db, run.id)),
        "waits": len(await _open_waits(db, run)),
        "decisions": len((await db.scalars(select(OrchestrationAuthorityDecision).where(
            OrchestrationAuthorityDecision.run_id == run.id))).all()),
        "meetings": len((await db.scalars(select(Meeting))).all()),
        "graph_runs": len((await db.scalars(select(GraphRun))).all()),
    }


async def _last_accepted_decision(db, run):
    return (await db.scalars(
        select(OrchestrationDecision).where(
            OrchestrationDecision.run_id == run.id, OrchestrationDecision.validator_status == "accepted",
        ).order_by(OrchestrationDecision.created_at, OrchestrationDecision.id)
    )).all()[-1]


# 1. Fresh authorized run ----------------------------------------------------
async def test_scenario_1_fresh_authorized_run_acts_then_waits_and_replays_idempotently(
    db_session, test_project, stub_decision
):
    planner = _agent("planner", ["planning"])
    db_session.add(planner)
    await db_session.flush()
    service, goal, run = await _authorized_run(db_session, test_project)
    before = await _snapshot(db_session, run)
    calls: list = []
    stub_decision(_plan_then_wait_stub(calls))

    result = await service.tick(db_session, run.id)

    assert await _assert_liveness(db_session, goal, run, before) == "action"
    assert run.plan_state["status"] == "requested"
    waits = await _open_waits(db_session, run)
    assert len(waits) == 1 and await _liveness_clause(db_session, goal, run, await _snapshot(db_session, run)) == "wait"
    settled = await _effect_counts(db_session, run)
    assert settled["action_types"].count("request_plan") == 1 and settled["tasks"] == 1

    # Replay 1: the same accepted decision is dispatched again.
    decision = await _last_accepted_decision(db_session, run)
    await service._dispatch_execution_decision(db_session, run, decision)
    assert await _effect_counts(db_session, run) == settled
    # Replay 2: an immediate idle tick (bounded wait still open) adds nothing and calls no model.
    seen = len(calls)
    idle = await service.tick(db_session, run.id)
    assert idle["authorized_execution"]["outcome"] == "waiting" and len(calls) == seen
    assert await _effect_counts(db_session, run) == settled
    assert await _liveness_clause(db_session, goal, run, await _snapshot(db_session, run)) == "wait"
    assert len(result["authorized_execution"]["action_ids"]) == 2


# 3a. Recoverable blocked task: missing agent + alternate -> reassign ---------
async def test_scenario_3a_blocked_task_with_missing_agent_reassigns_not_asks(
    db_session, test_project, monkeypatch
):
    monkeypatch.setattr("huddleroom.services.orchestration_service.TaskService.run", _fake_task_run([]))
    original = _recovery_agent("original", "developer", ["implementation"], is_active=False)
    alternate = _recovery_agent("alternate", "developer", ["implementation"])
    db_session.add_all([original, alternate])
    await db_session.flush()
    service, goal, run, task, _ = await _blocked_setup(db_session, test_project, agent=original, reason="no owner")
    before = await _snapshot(db_session, run)

    assert await service.recover_run(db_session, run.id, baseline_ready=True) == 1

    await _assert_liveness(db_session, goal, run, before, expect="action")
    await db_session.refresh(task)
    assert task.assigned_to == alternate.id
    assert [a.status for a in await _recovery_actions(db_session, run.id, "reassign_task")] == ["completed"]
    assert await _recovery_actions(db_session, run.id, "ask_human") == []
    # Replay: the reassigned (still blocked, fresh live session) task now waits on that session; nothing is
    # reassigned or asked twice, and the next replay is a pure no-op.
    await service.recover_run(db_session, run.id, baseline_ready=True)
    assert len(await _recovery_actions(db_session, run.id, "reassign_task")) == 1
    assert await _recovery_actions(db_session, run.id, "ask_human") == []
    settled = await _effect_counts(db_session, run)
    assert await service.recover_run(db_session, run.id, baseline_ready=True) == 0
    assert await _effect_counts(db_session, run) == settled


# 3b. Recoverable blocked task: live dependency -> bounded wait, no ask --------
async def test_scenario_3b_blocked_task_with_live_dependency_waits_without_asking(db_session, test_project):
    service, goal, run, task, agent = await _blocked_setup(db_session, test_project, reason="needs upstream")
    dep = await _make_orchestrated_task(
        db_session, test_project.id, run.id, (await _make_gate(db_session, run.id)).id, agent.id, status="in_progress",
    )
    task.depends_on = [str(dep.id)]
    await db_session.flush()
    before = await _snapshot(db_session, run)

    assert await service.recover_run(db_session, run.id, baseline_ready=True) == 1

    await _assert_liveness(db_session, goal, run, before, expect="wait")
    (wait,) = await _task_waits(db_session, run.id)
    assert str(dep.id) in wait.fallback["expected_result"] and wait.owner["task_id"] == str(task.id)
    assert await _recovery_actions(db_session, run.id, "ask_human") == []
    assert goal.status == "active" and run.status == "running"
    settled = await _effect_counts(db_session, run)
    assert await service.recover_run(db_session, run.id, baseline_ready=True) == 0
    assert await _effect_counts(db_session, run) == settled


# 4. Exact human authority request --------------------------------------------
async def test_scenario_4_decision_path_ask_human_creates_one_exact_authority_decision(
    db_session, test_project, stub_decision
):
    from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision

    service, goal, run = await _authorized_run(db_session, test_project)
    question = "Which payment provider should the checkout use, Stripe or Adyen?"
    stub_decision(lambda _ctx: {
        "action_type": "ask_human", "question": question, "reason": "Only the owner can pick the vendor.",
    })
    before = await _snapshot(db_session, run)

    await service.tick(db_session, run.id)

    await _assert_liveness(db_session, goal, run, before, expect="human")
    (decision,) = (await db_session.scalars(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.run_id == run.id))).all()
    assert decision.status == "pending" and decision.authority == "human" and decision.question == question
    settled = await _effect_counts(db_session, run)
    assert settled["decisions"] == 1 and settled["action_types"].count("ask_human") >= 1

    # Replays: same decision re-dispatched, and a further tick while blocked.
    await service._dispatch_execution_decision(db_session, run, await _last_accepted_decision(db_session, run))
    await service.tick(db_session, run.id)
    assert (await _effect_counts(db_session, run))["decisions"] == 1
    assert len([a for a in await _run_actions(db_session, run.id) if a.action_type == "ask_human"]) == 1
    assert await _liveness_clause(db_session, goal, run, await _snapshot(db_session, run)) == "human"


async def test_scenario_4b_owner_only_blocked_task_asks_the_exact_question_once(db_session, test_project):
    from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision

    service, goal, run, _task, _ = await _blocked_setup(db_session, test_project, reason="Need the prod API key")
    before = await _snapshot(db_session, run)

    assert await service.recover_run(db_session, run.id, baseline_ready=True) == 1

    await _assert_liveness(db_session, goal, run, before, expect="human")
    (decision,) = (await db_session.scalars(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.run_id == run.id))).all()
    assert decision.question == (
        'Task "Implementation work" is blocked: Need the prod API key. '
        "What input or decision do you need to supply to unblock it?"
    )
    settled = await _effect_counts(db_session, run)
    assert await service.recover_run(db_session, run.id, baseline_ready=True) == 0
    assert await _effect_counts(db_session, run) == settled



# 2. Active meeting + independent work -----------------------------------------
async def _accepted_plan_run_with_active_meeting(db, project):
    """Authorized run, accepted plan (its only item already released), one in-flight task owned by an
    ACTIVE meeting, plus an independent open commitment from a concluded meeting on a done task."""
    from huddleroom.models.meeting import Meeting, MeetingActionItem
    from huddleroom.models.task import Task

    service, run, parent, agent, _report = await _completed_parent_task(db, project)
    await _seed_accepted_plan(
        db, project, service, run, agent,
        plan_items=[{
            "id": "released-item", "work_function": "implementation", "scope": "Already released.",
            "deliverable": "Nothing new to release.", "agent_id": str(agent.id),
        }],
    )
    await service.reserve_action(
        db, run_id=run.id, idempotency_key=f"run:{run.id}:kind:release_item:released-item",
        action_type="release_item", request={"plan_item_id": "released-item"},
    )
    source = Task(
        project_id=project.id, title="Meeting-owned work", status="in_progress", assigned_to=agent.id,
        metadata_={"orchestration": {"run_id": str(run.id)}},
    )
    db.add(source)
    await db.flush()
    active = Meeting(project_id=project.id, title="Design sync", meeting_type="standup",
                     status="active", source_task_id=source.id)
    done_meeting = Meeting(project_id=project.id, title="Retro", meeting_type="standup",
                           status="completed", source_task_id=parent.id)
    db.add_all([active, done_meeting])
    await db.flush()
    item = MeetingActionItem(meeting_id=done_meeting.id, description="Publish the migration notes")
    db.add(item)
    await db.flush()
    return service, run, agent, source, active, item


async def test_scenario_2_active_meeting_does_not_suppress_independent_work(
    db_session, test_project, stub_decision
):
    service, run, agent, source, meeting, item = await _accepted_plan_run_with_active_meeting(db_session, test_project)
    goal = await db_session.get(OrchestrationGoal, run.goal_id)
    calls: list = []

    def decide(ctx):
        calls.append(ctx)
        if len(calls) == 1:
            assert any(f["id"] == str(item.id) for f in ctx["untracked_follow_ups"]), "independent work must be offered"
        return {
            "action_type": "create_delegation_task", "agent_id": str(agent.id), "work_function": "documentation",
            "scope": "Publish the migration notes", "deliverable": "Notes published.",
            "inputs": [f"meeting_action_item:{item.id}"],
        } if len(calls) == 1 else {
            "action_type": "noop", "reason": "Delegated; waiting", "wake_when": dict(PLAN_WAKE),
        }

    stub_decision(decide)
    before = await _snapshot(db_session, run)

    result = await service.tick(db_session, run.id)

    assert result["authorized_execution"]["step"] != "durable_source_active"
    await _assert_liveness(db_session, goal, run, before, expect="action")
    delegations = [a for a in await _run_actions(db_session, run.id)
                   if a.action_type == "create_delegation_task" and f"meeting_action_item:{item.id}" in (a.request.get("inputs") or [])]
    assert len(delegations) == 1 and delegations[0].status == "completed"
    # The active meeting itself was left alone (still active, same single source task).
    await db_session.refresh(meeting)
    assert meeting.status == "active"
    settled = await _effect_counts(db_session, run)

    # Replays: re-dispatching the delegation decision and ticking again create no second task/action.
    delegate_decision = next(d for d in await db_session.scalars(select(OrchestrationDecision).where(
        OrchestrationDecision.run_id == run.id)) if (d.parsed_decision or {}).get("action_type") == "create_delegation_task")
    await service._dispatch_execution_decision(db_session, run, delegate_decision)
    assert (await _effect_counts(db_session, run))["tasks"] == settled["tasks"]
    assert len([t for t in await _tasks_for_run(db_session, run.id) if t.title != source.title
                and (t.metadata_ or {}).get("orchestration", {}).get("action_id") == str(delegations[0].id)]) == 1
    await db_session.refresh(item)
    assert item.task_id is not None and item.status == "task_created"


# 5. Answered decision amid unrelated newer decisions ---------------------------
async def test_scenario_5_answered_decision_stays_follow_up_until_action_applies_it(
    db_session, test_project, stub_decision
):
    from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
    from huddleroom.services.orchestration_progress_view import OrchestrationProgressView

    planner = _agent("planner", ["planning"])
    db_session.add(planner)
    await db_session.flush()
    service, goal, run = await _authorized_run(db_session, test_project)
    anchor = OrchestrationAction(
        run_id=run.id, idempotency_key=f"anchor:{uuid.uuid4()}", action_type="ask_human", status="completed",
        request={"question": "Ship the beta?"},
    )
    db_session.add(anchor)
    await db_session.flush()
    answered = OrchestrationAuthorityDecision(
        goal_id=goal.id, run_id=run.id, decision_key=f"key-{uuid.uuid4()}", title="Ship the beta?",
        status="answered", authority="human", question="Ship the beta?", selected_option="ship",
        related_action_id=anchor.id, decided_at=_utcnow(),
    )
    # A newer, unrelated (cancelled) question must not hide the answer; a pending one would instead put the run
    # on a local authority wait and skip the decision path entirely, which is the human arm of the invariant.
    unrelated = OrchestrationAuthorityDecision(
        goal_id=goal.id, run_id=run.id, decision_key=f"key-{uuid.uuid4()}", title="Unrelated vendor choice",
        status="cancelled", authority="human", question="Which vendor?", options=[{"key": "a"}],
        related_action_id=anchor.id,
    )
    db_session.add_all([answered, unrelated])
    await db_session.flush()

    async def follow_up_ids():
        situation = await OrchestrationProgressView().build(db_session, goal, run)
        return [f["id"] for f in situation.untracked_follow_ups if f["kind"] == "answered_decision"]

    assert await follow_up_ids() == [str(answered.id)]
    mode = {"step": 0}

    def decide(ctx):
        mode["step"] += 1
        if mode["step"] == 1:
            # A noop that merely names the decision does not consume it.
            return {"action_type": "noop", "reason": "Considering the answer", "wake_when": dict(PLAN_WAKE),
                    "applies_decision_id": str(answered.id)}
        if mode["step"] == 2:
            return {**_request_plan(ctx), "applies_decision_id": str(answered.id)}
        return {"action_type": "noop", "reason": "Plan task in flight", "wake_when": dict(PLAN_WAKE)}

    stub_decision(decide)

    before = await _snapshot(db_session, run)
    await service.tick(db_session, run.id)
    await _assert_liveness(db_session, goal, run, before, expect="wait")
    assert len(await _open_waits(db_session, run, ORCHESTRATOR_WAIT_OWNER_TYPE)) == 1
    assert await follow_up_ids() == [str(answered.id)], "noop/wait/newer decisions must not consume the answer"

    for wait in await _open_waits(db_session, run):
        wait.status = "cleared"
    await db_session.flush()
    before = await _snapshot(db_session, run)
    await service.tick(db_session, run.id)

    await _assert_liveness(db_session, goal, run, before, expect="action")
    assert await follow_up_ids() == [], "a completed action carrying applies_decision_id consumes the answer"
    plan_actions = [a for a in await _run_actions(db_session, run.id) if a.action_type == "request_plan"]
    assert len(plan_actions) == 1
    assert plan_actions[0].dispatch_contract["applies_decision_id"] == str(answered.id)
    settled = await _effect_counts(db_session, run)

    # Replay: re-dispatching the applying decision neither duplicates the action nor resurrects the follow-up.
    applying = next(d for d in await db_session.scalars(select(OrchestrationDecision).where(
        OrchestrationDecision.run_id == run.id)) if (d.parsed_decision or {}).get("action_type") == "request_plan")
    await service._dispatch_execution_decision(db_session, run, applying)
    assert (await _effect_counts(db_session, run))["action_types"].count("request_plan") == 1
    assert (await _effect_counts(db_session, run))["tasks"] == settled["tasks"]
    assert await follow_up_ids() == []


# 6. Failed then successful verification ---------------------------------------
async def test_scenario_6_failed_then_successful_verification_needs_no_attention(db_session, test_project):
    from huddleroom.models.orchestration import OrchestrationGate
    from huddleroom.services.orchestration_progress_view import OrchestrationProgressView

    goal, run = await _bare_supervised_run(db_session, test_project)
    run.plan_state = {"status": "accepted"}
    gate = OrchestrationGate(run_id=run.id, success_criterion_key="plan_item:x", gate_type="work_completed",
                             required_evidence={}, status="open")
    db_session.add(gate)
    await db_session.flush()
    service = OrchestrationService()
    request = {"action_type": "request_verification", "gate_id": str(gate.id), "work_function": "validation"}
    key = f"run:{run.id}:kind:verify"
    with pytest.raises(Exception):  # no producer task: the executor fails the action deterministically
        await service.execute_request_verification_action(db_session, run.id, request, key)
    first = await service._existing_action_for_key(db_session, run.id, key)
    assert first.status == "failed"

    async def gate_follow_ups():
        situation = await OrchestrationProgressView().build(db_session, goal, run)
        return [f for f in situation.untracked_follow_ups if f["kind"] == "unverified_gate"]

    before = await _snapshot(db_session, run)
    (follow_up,) = await gate_follow_ups()
    assert follow_up["suggested_actions"] == ["request_verification"]
    result = await service.supervision.reconcile_local(db_session, goal, run, allow_release=False)
    assert result["outcome"] != "needs_attention" and not run.active_blockers
    assert await _liveness_clause(db_session, goal, run, before) in {"wait", "action", "human"} or result["outcome"] == "continue"

    # Retry uses an attempt-suffixed key; success (completed attempt + accepted gate) clears everything.
    suffix = await service.verification_attempt_suffix(db_session, run.id, gate.id)
    assert suffix == ":attempt:1"
    retry = await service.reserve_action(
        db_session, run_id=run.id, idempotency_key=key + suffix, action_type="request_verification", request=request,
    )
    retry = await service._mark_action_completed(db_session, retry, target_type="task", target_id=uuid.uuid4())
    gate.status = "accepted"
    await db_session.flush()
    assert await gate_follow_ups() == []
    settled = await _effect_counts(db_session, run)
    again = await service.reserve_action(
        db_session, run_id=run.id, idempotency_key=key + suffix, action_type="request_verification", request=request,
    )
    assert again.id == retry.id and await _effect_counts(db_session, run) == settled
    after = await service.supervision.reconcile_local(db_session, goal, run, allow_release=False)
    assert after["outcome"] != "needs_attention" and not run.active_blockers


# 7. Repeated no-progress wake ---------------------------------------------------
async def test_scenario_7_repeated_no_progress_wake_asks_after_limit_and_reasks_after_answer(
    db_session, test_project, test_agent, monkeypatch
):
    from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision

    goal, run = await _idle_setup(db_session, test_project, test_agent)
    _situation(monkeypatch, untracked_follow_ups=[STUCK])
    before = await _snapshot(db_session, run)

    results = await _ticks(db_session, goal, run, 4)

    assert [r["outcome"] for r in results] == ["continue"] * 3 + ["needs_attention"]
    (first,) = await _asks(db_session, run)
    assert "Gate G awaiting verification" in first.request["question"]
    assert await _liveness_clause(db_session, goal, run, before) == "human"
    settled = await _effect_counts(db_session, run)
    # Replay while the question is pending: no second ask, no new decision.
    await _ticks(db_session, goal, run, 2)
    # (a due authority-decision wait may add its own attention warning; the ask and the decision must not repeat)
    after = await _effect_counts(db_session, run)
    assert after["decisions"] == settled["decisions"] and after["action_types"].count("ask_human") == 1

    decision = await db_session.get(OrchestrationAuthorityDecision, first.target_id)
    decision.status = "answered"
    await db_session.flush()
    before_second = await _snapshot(db_session, run)
    results = await _ticks(db_session, goal, run, 4)
    assert [r["outcome"] for r in results] == ["continue"] * 3 + ["needs_attention"]
    asks = await _asks(db_session, run)
    assert len(asks) == 2 and asks[0].idempotency_key != asks[1].idempotency_key
    assert await _liveness_clause(db_session, goal, run, before_second) == "human"


# 8. Expired analyzer claim ------------------------------------------------------
async def test_scenario_8_expired_analyzer_claim_is_reclaimed_and_leaves_a_bounded_wait(
    db_session, test_project, monkeypatch
):
    from huddleroom.services.orchestration_supervision_scheduler import OrchestrationSupervisionScheduler

    _goal, run = await _bare_supervised_run(db_session, test_project)
    run.supervision_state = _live_claim_state(expires=_NOW - timedelta(seconds=1))
    sessions = await _fresh_sessions(db_session, monkeypatch)
    calls = []

    async def judge(_payload):
        calls.append(1)
        return _provider_assessment()

    scheduler = OrchestrationSupervisionScheduler(judge=judge)
    async with sessions() as worker:
        assert await scheduler.evaluate_run(worker, run.id, now=_NOW) == 1
    assert calls == [1]
    state = await _persisted_state(sessions, run.id)
    assert state["judgment_in_flight"] is False and "judgment_claim_token" not in state
    async with sessions() as observer:
        waits = (await observer.scalars(select(OrchestrationWait).where(
            OrchestrationWait.run_id == run.id, OrchestrationWait.status == "open"))).all()
        actions = (await observer.scalars(select(OrchestrationAction).where(OrchestrationAction.run_id == run.id))).all()
    assert len(waits) >= 1 and all(w.fallback["action_type"] in {"continue", "attention"} for w in waits)
    assert len(actions) == 1
    # Replay: the finished judgment is not due again, so nothing is re-judged or duplicated.
    async with sessions() as worker:
        await scheduler.evaluate_run(worker, run.id, now=_NOW)
    async with sessions() as observer:
        assert len((await observer.scalars(select(OrchestrationAction).where(
            OrchestrationAction.run_id == run.id))).all()) == 1
        assert len((await observer.scalars(select(OrchestrationWait).where(
            OrchestrationWait.run_id == run.id, OrchestrationWait.status == "open"))).all()) == len(waits)
    assert calls == [1]


# 9. Analyzer failure threshold --------------------------------------------------
async def test_scenario_9_three_analyzer_failures_warn_and_block_once_and_success_clears(
    db_session, test_project, monkeypatch
):
    from huddleroom.models.orchestration_process import OrchestrationWarning
    from huddleroom.services.orchestration_supervision_scheduler import OrchestrationSupervisionScheduler

    _goal, run = await _bare_supervised_run(db_session, test_project)
    run.supervision_state = {"needs_judgment": True, "judgment_due_at": _NOW.isoformat(), "judgment_failures": 1}
    sessions = await _fresh_sessions(db_session, monkeypatch)
    clock = _fake_clock(monkeypatch)

    async def fail(_payload):
        raise RuntimeError("down")

    async def ok(_payload):
        return _provider_assessment()

    async def evaluate(judge):
        async with sessions() as worker:
            await OrchestrationSupervisionScheduler(judge=judge).evaluate_run(worker, run.id, now=clock["t"])

    async def snapshot():
        async with sessions() as observer:
            current = await observer.get(OrchestrationRun, run.id)
            warnings = (await observer.scalars(select(OrchestrationWarning).where(
                OrchestrationWarning.goal_id == run.goal_id,
                OrchestrationWarning.warning_type == "supervision_judgment_failures"))).all()
            blockers = [b for b in current.active_blockers if b.get("kind") == "supervision_judgment_failures"]
            return dict(current.supervision_state), warnings, blockers

    await evaluate(fail)
    state, warnings, blockers = await snapshot()
    assert state["judgment_failures"] == 3 and len(warnings) == 1 and len(blockers) == 1
    assert blockers[0]["failure_count"] == 3 and blockers[0]["next_retry_at"] > clock["t"].isoformat()
    await evaluate(fail)  # replay inside the backoff: nothing new
    _state, warnings, blockers = await snapshot()
    assert len(warnings) == 1 and len(blockers) == 1
    clock["t"] += timedelta(hours=1)
    await evaluate(ok)
    state, warnings, blockers = await snapshot()
    assert state["judgment_failures"] == 0 and blockers == [] and [w.active for w in warnings] == [False]


# 10. Unfinished meeting commitment at closeout -----------------------------------
async def test_scenario_10_unfinished_meeting_commitment_blocks_closeout_until_resolved(
    db_session, test_project, completion_ready_goal, safe_effectiveness_review_continue  # noqa: F811
):
    from fastapi import HTTPException

    from huddleroom.models.meeting import Meeting, MeetingActionItem
    from huddleroom.models.task import Task
    from huddleroom.services.orchestration_progress_view import OrchestrationProgressView

    service, goal, run = completion_ready_goal
    source = Task(project_id=test_project.id, title="Source", status="done",
                  metadata_={"orchestration": {"run_id": str(run.id)}})
    db_session.add(source)
    await db_session.flush()
    meeting = Meeting(project_id=test_project.id, title="Sync", meeting_type="standup", source_task_id=source.id)
    db_session.add(meeting)
    await db_session.flush()
    item = MeetingActionItem(meeting_id=meeting.id, description="Unhandled commitment")
    db_session.add(item)
    await db_session.flush()

    for _ in range(2):  # replay: identical rejection, no side effects
        with pytest.raises(HTTPException) as exc:
            await service._closeout_preconditions_manifest(db_session, goal, run)
        assert exc.value.status_code == 409 and str(item.id) in exc.value.detail
    settled = await _effect_counts(db_session, run)
    await service.tick(db_session, run.id)
    assert run.status != "completed"
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    assert str(item.id) in {f["id"] for f in situation.untracked_follow_ups}
    before = await _snapshot(db_session, run)
    assert before is not None and (await _effect_counts(db_session, run))["decisions"] == settled["decisions"]

    item.status = "waived"
    await db_session.flush()
    manifest = await service._closeout_preconditions_manifest(db_session, goal, run)
    assert manifest["unresolved_meeting_commitments"] == []
