"""Integrated three-level planning hierarchy, no network.

Scenario 1: Auto root Roadmap P (stubbed classifier) -> nested Roadmap C (parent_explicit) -> leaf Outcome g1
(parent_auto). The leaf planner and two independent audits share the root's two slots; audit-1's first
attempt fails and is retried before the plan adopts both audits and joins them in a synthesis item; C replans
safely (v2) while a probe task runs, preserving g1's accepted evidence; g2 releases from v2; C then P close
only through independent integration. Scenario 2: explicit types are never classified, and an Auto root
corrected after Start needs its own baseline and fresh Start. Scenario 3: a failed predecessor attempt is retried; its gate reopens on the successful attempt and the dependent releases.

Shortcuts (documented): LLM decisions are scripted per run through the real dispatcher; agent sessions are
finished by writing the terminal Session row + events the adapters would write (the local runner hand-off
is recorded, never executed); provider, subprocess, HTTP and socket entry points fail the test if touched;
Roadmap goals carry an explicit `team.agent_ids` context; verification and the C replan request are issued
through their real service actions rather than an LLM decision; one 300s decision backstop is made due
instead of sleeping; depth/count limits are checked through the roadmap service against the frozen policy.
"""

import ast
import asyncio
import json
import socket
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import httpx
import litellm
import pytest
from sqlalchemy import func, select

from huddleroom.config import settings
from huddleroom.models.agent import Agent
from huddleroom.models.artifact import Artifact
from huddleroom.models.orchestration import (
    OrchestrationAction, OrchestrationBudgetReservation, OrchestrationEvidence, OrchestrationGate,
    OrchestrationGoal, OrchestrationRoadmapItem, OrchestrationRoadmapVersion, OrchestrationRun,
    OrchestrationWait, compute_goal_contract_fingerprint,
)
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.schemas.orchestration import (
    GoalShapeDecision, GoalTypeCorrectionApplyRequest, GoalTypeCorrectionPreviewRequest, OrchestrationGoalCreate,
)
from huddleroom.services import orchestration_completion
from huddleroom.services.orchestration_authority_service import (
    OrchestrationAuthorityDecisionService, effective_authorization,
)
from huddleroom.services.orchestration_dependencies import evidence_semantics
from huddleroom.services.orchestration_goal_shape import OrchestrationGoalShapeService
from huddleroom.services.orchestration_plan_ingestion import plan_artifact_id
from huddleroom.services.orchestration_roadmap_service import OrchestrationRoadmapService
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.event_bus import emit_event_once
from huddleroom.services.session_sync import sync_task_from_session
from huddleroom.workers import task_runner
from tests.conftest import bind_plan_author_team, complete_baseline_processes, heal_baseline_drift_for_test
from tests.test_orchestration_roadmap_children import goal_item
from tests.test_orchestration_roadmap_e2e import (
    _assert_budget_conserved, _children_of, _close_roadmap, _item, _run_of, _snapshot, _tick_until,
)
from tests.test_orchestration_roadmap_integration import (
    _child_actors, _complete_child, _complete_roadmap_task, _integration_gate, _record_measured_completion,
)
from tests.test_orchestration_roadmap_task_items import roadmap_task
from tests.test_orchestration_runtime_e2e import (
    _agent, _authorized_run, _run_actions, _seed_accepted_plan, _tasks_for_run,
)

ROOT_OBJECTIVE = "Launch v2: core platform area, then rollout."


# --- harness ---------------------------------------------------------------------------------

@pytest.fixture
def external_calls(monkeypatch):
    """Every network/provider/CLI entry point records and fails; the test asserts the list stays empty."""
    calls, dispatched = [], []

    def forbid(name):
        def fail(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"external call attempted: {name}")

        async def afail(*args, **kwargs):
            fail(*args, **kwargs)
        return fail, afail

    for target, attr, is_async in (
        (litellm, "acompletion", True), (litellm, "completion", False),
        (orchestration_completion, "orchestration_completion", True),
        (orchestration_completion, "_run", True),
        (asyncio, "create_subprocess_exec", True), (asyncio, "create_subprocess_shell", True),
        (httpx.AsyncClient, "send", True), (httpx.Client, "send", False),
        (socket.socket, "connect", False), (socket.socket, "connect_ex", False),
    ):
        sync, async_ = forbid(f"{getattr(target, '__name__', target)}.{attr}")
        monkeypatch.setattr(target, attr, async_ if is_async else sync)
    # The local runner hand-off (after commit) is recorded, never executed: the test plays the adapter.
    monkeypatch.setattr(task_runner, "register_session", lambda *args, **kwargs: dispatched.append(args[0]))
    return calls


@pytest.fixture
def classifier():
    """Stub the Auto shape model call (the conftest stub's seam) with a per-objective script."""
    shapes = {ROOT_OBJECTIVE: ("roadmap", "roadmap")}
    calls = []

    async def classify(self, project_id, data):
        calls.append(data.objective)
        goal_type, mode = shapes.get(data.objective, ("outcome", "delivery"))
        return GoalShapeDecision(
            goal_type=goal_type, goal_type_source="auto", goal_type_rationale=f"Stub: {data.objective} is {goal_type}.",
            planning_mode=mode, planning_rationale=f"Stub: {mode} planning.",
        )

    with patch.object(OrchestrationGoalShapeService, "_classify", classify):
        yield shapes, calls


@pytest.fixture
def auto_start(monkeypatch):
    """Task activation is off (manual completion drives sessions) unless a test switches it on."""
    original = OrchestrationService._start_released_tasks
    switch = {"on": False}

    async def maybe_start(self, db, goal, run):
        return await original(self, db, goal, run) if switch["on"] else 0

    monkeypatch.setattr(OrchestrationService, "_start_released_tasks", maybe_start)
    return switch


def _script(monkeypatch, service, scripts):
    """Scripted LLM decision per run id (`scripts[run_id](run)`; None/missing = noop) through the real dispatcher."""
    async def request(db, run_id, adapter=None):
        run = await db.get(OrchestrationRun, run_id)
        goal = await db.get(OrchestrationGoal, run.goal_id)
        context = await service._decision_context(db, goal, run)
        script = scripts.get(run_id)
        decision = (script(run) if script else None) or {"action_type": "noop", "reason": "Events drive E2E."}
        return await service.record_validated_decision(db, run_id, context, {}, decision)

    monkeypatch.setattr(service, "request_llm_decision", request)


async def _count(db, model, *where):
    return await db.scalar(select(func.count()).select_from(model).where(*where))


async def _team_context(db, goal):
    # Shortcut: an explicit team list (as in the Roadmap E2E) instead of a re-run hierarchy process.
    agent_ids = [str(agent_id) for agent_id in await db.scalars(select(Agent.id).where(Agent.is_active.is_(True)))]
    goal.orchestrator_context = {**(goal.orchestrator_context or {}), "team": {"agent_ids": agent_ids}}
    await db.flush()


async def _plan_roadmap_by_tick(db, project, service, scripts, goal, run, planner, items, summary):
    """request_plan decision -> planner envelope -> human approval -> accept_plan, all through tick."""
    def plan(current):
        state = current.plan_state or {}
        if not state.get("planning_task_id"):
            return {"action_type": "request_plan", "agent_id": str(planner.id), "work_function": "planning",
                    "scope": f"Plan the {goal.objective} Roadmap."}
        if state.get("plan_artifact_id") and state.get("status") != "accepted":
            return {"action_type": "accept_plan", "plan_artifact_id": state["plan_artifact_id"]}
        return None

    scripts[run.id] = plan
    await service.tick(db, run.id)
    task = await db.get(Task, uuid.UUID(run.plan_state["planning_task_id"]))
    assert task.metadata_["orchestration_contract"]["report_schema"]["plan_kind"] == "roadmap"
    await _record_measured_completion(db, project.id, run, task, task.assigned_to,
                                      json.dumps(_roadmap_envelope(items, summary)))
    await service.tick(db, run.id)  # ingests the envelope and opens the human plan approval
    decision = await db.scalar(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.run_id == run.id, OrchestrationAuthorityDecision.status == "pending",
        OrchestrationAuthorityDecision.decision_key.like("roadmap_plan:%")))
    assert decision is not None, run.plan_state
    await OrchestrationAuthorityDecisionService().answer_decision(
        db, decision, selected_option="approve", decided_by_user_id=goal.manager_user_id)
    await _tick_until(db, service, run, lambda: (run.plan_state or {}).get("status") == "accepted")
    scripts.pop(run.id)
    await db.refresh(run)
    return task


def _roadmap_envelope(items, summary):
    return {"schema_version": 1, "plan_kind": "roadmap", "summary": summary,
            "plan_items": [{**item, "why_runnable": item.get("why_runnable", "Ready when its deps accept.")}
                           for item in items]}


def _report(evidence):
    """The canonical work report (strict schema) an audit returns."""
    return json.dumps({"status": "done", "changes": [], "evidence": [evidence], "criterion_progress": {},
                       "decisions": [], "risks": [], "open_questions": [], "next_step": "Synthesize.",
                       "collaboration_need": None})


async def _live_count(db):
    return await _count(db, Task, Task.status == "in_progress")


async def _audits(db, service, run):
    tasks = {service._task_work_function(task): task for task in await _tasks_for_run(db, run.id)}
    return tasks["investigation"], tasks["review"]


async def _item_gate(db, run, item_id):
    return await db.scalar(select(OrchestrationGate).where(
        OrchestrationGate.run_id == run.id, OrchestrationGate.success_criterion_key == f"plan_item:{item_id}"))


async def _verify(db, project, service, run, gate, verifier):
    """Independent verification of one gate through the real request_verification action."""
    action = await service.execute_request_verification_action(
        db, run.id, {"action_type": "request_verification", "gate_id": str(gate.id), "work_function": "validation"},
        f"run:{run.id}:kind:request_verification:{gate.success_criterion_key}")
    task = await db.get(Task, action.target_id)
    assert task.assigned_to == verifier.id
    await service.tick(db, run.id)
    await db.refresh(task)
    output = json.dumps({"status": "done", "verdict": "accepted", "evidence": [f"independent check {gate.id}"]})
    if task.status == "in_progress":
        await _finish(db, project, run, task, output)
    else:
        await _record_measured_completion(db, project.id, run, task, verifier.id, output)
    await service.tick(db, run.id)
    await db.refresh(gate)
    assert gate.status == "accepted", gate.failure_reason
    return task


async def _synthesis(db, service, run):
    return next((task for task in await _tasks_for_run(db, run.id)
                 if service._task_work_function(task) == "implementation"), None)


async def _tick_until_task(db, service, run, work_function):
    for _ in range(4):
        task = next((task for task in await _tasks_for_run(db, run.id)
                     if service._task_work_function(task) == work_function and task.status == "in_progress"), None)
        if task is not None:
            return task
        await service.tick(db, run.id)
    raise AssertionError(f"no live {work_function} task")


async def _assert_replay_adds_nothing(db, service, models, runs):
    before = await _snapshot(db, models)
    for run in runs:
        await service.tick(db, run.id)
    after = await _snapshot(db, models)
    assert {m: [r["id"] for r in rows] for m, rows in after.items()} == {
        m: [r["id"] for r in rows] for m, rows in before.items()}


async def _elapse_backstops(db, run):
    """Shortcut for wall-clock time: the scripted noop left a 300s decision backstop; make it due now."""
    for wait in await db.scalars(select(OrchestrationWait).where(
            OrchestrationWait.run_id == run.id, OrchestrationWait.status == "open")):
        wait.due_recheck_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    await db.flush()


async def _finish(db, project, run, task, output, *, failed=False):
    """Play the adapter for the task's live session: terminal status, output, measured usage, events."""
    session = await db.scalar(select(Session).where(
        Session.task_id == task.id, Session.status.in_(("pending", "running"))))
    assert session is not None, task.title
    session.status, session.output = ("failed", None) if failed else ("completed", output)
    session.metadata_ = {**session.metadata_, "token_count_in": 1, "token_count_out": 1,
                         **({"attempt": {"effect_state": "started"}} if failed else {})}
    session.started_at, session.ended_at = run.started_at, run.started_at + timedelta(minutes=1)
    await sync_task_from_session(db, session)
    status = task.status
    await emit_event_once(db, project.id, "task.status_changed",
                          {"task_id": str(task.id), "status": status, "previous_status": "in_progress"},
                          dedup_key=f"hier-task-{status}:{session.id}")
    if not failed:
        await emit_event_once(db, project.id, "session.completed",
                              {"session_id": str(session.id), "task_id": str(task.id)},
                              dedup_key=f"hier-session:{session.id}")
    await db.flush()
    return session


# --- scenario 1: Auto root Roadmap -> nested Roadmap -> leaf Outcome ----------------------------

async def test_auto_root_roadmap_plans_nested_roadmap_and_leaf_outcome_without_network(
    db_session, test_project, monkeypatch, external_calls, classifier, auto_start,
    safe_effectiveness_review_continue,
):
    db, project = db_session, test_project
    shapes, classified = classifier
    monkeypatch.setattr(settings, "orchestration_max_actions_per_tick", 1)
    # Frozen into the root at creation: P(0) -> C(1) -> leaves(2), at most 3 descendants.
    monkeypatch.setattr(settings, "orchestration_max_goal_depth", 2)
    monkeypatch.setattr(settings, "orchestration_max_descendant_goals", 3)
    planner, producer, verifier, summarizer = await _child_actors(db)
    builder = _agent("hier-builder", ["implementation"])
    auditor = _agent("hier-auditor", ["investigation", "review"])
    db.add_all([builder, auditor])
    for agent in (planner, producer, verifier, summarizer, builder, auditor):
        # Small adapter limits: a live session commits only its own clamp, so two can share a leaf budget.
        agent.config = {"max_tokens": 30, "session_timeout_seconds": 300}
    await db.flush()
    service = OrchestrationService()
    scripts = {}
    _script(monkeypatch, service, scripts)

    # 1. Auto root: one classifier call resolves Roadmap; nothing else is classified.
    root, root_run = await service.create_goal(db, project.id, OrchestrationGoalCreate(
        objective=ROOT_OBJECTIVE, success_criteria=[{"key": "delivered", "description": "Roadmap is integrated."}],
        constraints={}, budget={"caps": {"max_tokens": 1000, "max_turns": 40, "max_hours": 10}},
    ), created_by_user_id=None)
    assert (root.goal_type, root.goal_type_source, root.planning_mode) == ("roadmap", "auto", "roadmap")
    assert (root_run.goal_type_snapshot, root_run.planning_mode_snapshot) == ("roadmap", "roadmap")
    assert root.orchestrator_context["expansion_policy"] == {
        "max_concurrent_tasks": 2, "max_goal_depth": 2, "max_descendant_goals": 3, "depth": 0}
    assert classified == [ROOT_OBJECTIVE]
    root_run.baseline_authorized = True
    await complete_baseline_processes(db, root, root_run)
    await service.tick(db, root_run.id)
    assert (root_run.phase, root_run.status) == ("ready", "running")
    await service.start_run(db, project.id, root.id, actor="human:e2e")
    assert root_run.phase == "authorized"
    await bind_plan_author_team(db, root, planner)
    await _team_context(db, root)
    root.weight = "trivial"  # keep the unrelated effectiveness analyzer deterministic
    await heal_baseline_drift_for_test(db, root, root_run)

    build = {**roadmap_task("build"), "agent_id": str(builder.id)}
    area = {**goal_item("area", allocation={"max_tokens": 600, "max_turns": 24, "max_hours": 4}),
            "goal_type": "roadmap", "depends_on": ["build"]}
    await _plan_roadmap_by_tick(db, project, service, scripts, root, root_run, planner, [build, area],
                                "Build first, then the nested platform area.")
    version = await OrchestrationRoadmapService(service).current_version(db, root.id)
    assert version is not None and version.run_id == root_run.id
    await _assert_budget_conserved(db, root)
    assert external_calls == []

    # 2. build accepted -> P releases only the nested Roadmap C (parent_explicit: no classifier call).
    assert (await service.tick(db, root_run.id))["authorized_execution"]["item_key"] == "build"
    build_row = await _item(db, root, "build")
    await _complete_roadmap_task(db, project.id, service, root_run, build_row, verifier.id)
    await service.tick(db, root_run.id)
    (area_goal,) = await _children_of(db, root)
    area_run = await _run_of(db, area_goal)
    assert (area_goal.goal_type, area_goal.goal_type_source, area_goal.planning_mode) == (
        "roadmap", "parent_explicit", "roadmap")
    assert area_run.phase == "authorized" and classified == [ROOT_OBJECTIVE]
    area_goal.weight = "trivial"
    await _assert_budget_conserved(db, root)

    g1 = {**goal_item("g1", allocation={"max_tokens": 250, "max_turns": 12, "max_hours": 1}), "goal_type": "auto"}
    probe = {**roadmap_task("probe", depends_on=["g1"]), "agent_id": str(builder.id)}
    g2 = {**goal_item("g2", allocation={"max_tokens": 150, "max_turns": 5, "max_hours": 1}),
          "goal_type": "outcome", "depends_on": ["probe"]}
    await _plan_roadmap_by_tick(db, project, service, scripts, area_goal, area_run, planner, [g1, probe, g2],
                                "Leaf g1, a probe task, then g2.")
    v1 = await OrchestrationRoadmapService(service).current_version(db, area_goal.id)
    assert (await service.tick(db, area_run.id))["authorized_execution"]["item_key"] == "g1"
    (leaf,) = await _children_of(db, area_goal)
    leaf_run = await _run_of(db, leaf)
    assert (leaf.goal_type, leaf.goal_type_source, leaf.planning_mode) == ("outcome", "parent_auto", "delivery")
    assert classified == [ROOT_OBJECTIVE, "g1"]
    leaf.weight = "trivial"
    for goal in (root, area_goal):
        await _assert_budget_conserved(db, goal)

    # 3. Leaf: the planner and two independent audits share the root's two slots.
    audits = []

    def leaf_script(current):
        state = current.plan_state or {}
        if not state.get("planning_task_id"):
            return {"action_type": "request_plan", "agent_id": str(planner.id), "work_function": "planning",
                    "scope": "Plan the g1 delivery."}
        if state.get("status") == "requested" and len(audits) < 2:
            n = len(audits)
            audits.append(n)
            return {"action_type": "create_delegation_task", "agent_id": str(auditor.id),
                    "work_function": ("investigation", "review")[n], "scope": f"Audit area {n} for g1.",
                    "deliverable": f"Audit findings {n}.", "decision_use": f"Choose the g1 design for area {n}.",
                    "success_evidence": [f"List of call sites in area {n}"],
                    "budget": {"max_tokens": 30, "max_turns": 2, "max_hours": 0.25}}
        if state.get("plan_artifact_id") and state.get("status") != "accepted":
            return {"action_type": "accept_plan", "plan_artifact_id": state["plan_artifact_id"]}
        return None

    scripts[leaf_run.id] = leaf_script
    auto_start["on"] = True
    for _ in range(3):
        await service.tick(db, leaf_run.id)
    planning = await db.get(Task, uuid.UUID(leaf_run.plan_state["planning_task_id"]))
    audit0, audit1 = await _audits(db, service, leaf_run)
    assert (planning.status, audit0.status, audit1.status) == ("in_progress", "in_progress", "backlog")
    assert await _live_count(db) == 2  # the root's two slots, shared by planner and audits
    assert external_calls == []

    # 4. audit-0 reports evidence; its slot goes to audit-1, whose first attempt fails (the failed predecessor).
    await _finish(db, project, leaf_run, audit0, _report("Area 0 call sites listed."))
    await service.tick(db, leaf_run.id)
    readiness = await db.get(OrchestrationGate, uuid.UUID(audit0.metadata_["orchestration"]["gate_id"]))
    assert readiness.status == "accepted" and evidence_semantics(
        readiness.gate_type, readiness.required_evidence, "task", "accepted", None,
    ) == {"evidence_role": "discovery", "review_stage": "preliminary", "acceptance_verdict": "accepted"}
    await db.refresh(audit1)
    assert audit1.status == "in_progress" and await _live_count(db) == 2
    await _finish(db, project, leaf_run, audit1, None, failed=True)
    await service.tick(db, leaf_run.id)
    retry = await _approved_retry(db, service, leaf_run, audit1)
    await db.refresh(audit1)
    assert retry.status == "completed" and audit1.status == "in_progress"
    await _finish(db, project, leaf_run, audit1, _report("Area 1 review notes."))
    await service.tick(db, leaf_run.id)

    # 5. The planner adopts both audits and joins them in a synthesis item.
    envelope = {"schema_version": 1, "plan_kind": "delivery", "summary": "Adopt both audits, then synthesize.",
                "plan_items": [
                    *[{"id": f"audit-{n}", "work_function": wf, "scope": f"Audit area {n} for g1.",
                       "deliverable": f"Audit findings {n}.", "why_runnable": "Already running.",
                       "adopt_task_id": str(task.id)} for n, (wf, task) in enumerate(
                           (("investigation", audit0), ("review", audit1)))],
                    {"id": "synthesis", "work_function": "implementation", "agent_id": str(builder.id),
                     "scope": "Implement g1 from both audits.", "deliverable": "Delivered g1.",
                     "why_runnable": "Both audits accepted.", "depends_on": ["audit-0", "audit-1"],
                     "success_criterion_keys": ["local"]}]}
    await _finish(db, project, leaf_run, planning, json.dumps(envelope))
    await _elapse_backstops(db, leaf_run)
    await _tick_until(db, service, leaf_run, lambda: leaf_run.plan_state.get("status") == "accepted")
    await service.tick(db, leaf_run.id)  # adopts both audits: no new task, no new slot
    audit_gates = [await _item_gate(db, leaf_run, f"audit-{n}") for n in range(2)]
    assert [task.metadata_["orchestration"]["plan_item_gate_id"] for task in (audit0, audit1)] == [
        str(gate.id) for gate in audit_gates]
    assert sorted(a.target_id for a in await _run_actions(db, leaf_run.id) if a.action_type == "adopt_task") == sorted(
        [audit0.id, audit1.id])
    assert len(await _tasks_for_run(db, leaf_run.id)) == 3  # planner + two audits; synthesis waits on the join

    # 6. The join: synthesis releases only after independent verification accepts BOTH adopted audit gates.
    await _verify(db, project, service, leaf_run, audit_gates[0], verifier)
    assert await _synthesis(db, service, leaf_run) is None
    await _verify(db, project, service, leaf_run, audit_gates[1], verifier)
    synthesis = await _synthesis(db, service, leaf_run)
    assert synthesis is not None and synthesis.assigned_to == builder.id
    await _tick_until(db, service, leaf_run, lambda: synthesis.status == "in_progress")

    # 7. Synthesis delivers, is verified, and the leaf closes through its final summary.
    await _finish(db, project, leaf_run, synthesis, _report("g1 delivered."))
    await service.tick(db, leaf_run.id)
    synthesis_gate = await _item_gate(db, leaf_run, "synthesis")
    await _verify(db, project, service, leaf_run, synthesis_gate, verifier)
    proof = await db.scalar(select(OrchestrationEvidence).where(
        OrchestrationEvidence.gate_id == synthesis_gate.id, OrchestrationEvidence.source_type == "verification",
        OrchestrationEvidence.verdict == "accepted"))
    summary = await _tick_until_task(db, service, leaf_run, "summarization")
    assert summary.assigned_to == summarizer.id
    await _finish(db, project, leaf_run, summary, json.dumps({
        "summary": "g1 delivered from both audits.", "unresolved_gaps": [],
        "criteria": [{"criterion_key": "local", "evidence_ids": [str(proof.id)]}]}))
    await _tick_until(db, service, leaf_run, lambda: leaf_run.status == "completed")
    await db.refresh(leaf)
    assert (leaf.status, leaf_run.phase) == ("completed", "completed")
    assert external_calls == []

    # 8. C settles g1: one child_goal evidence row on g1's gate; the probe task item releases next.
    g1_row = await _item(db, area_goal, "g1")
    await _tick_until(db, service, area_run, lambda: g1_row.completed_at is not None)
    g1_gate = await db.get(OrchestrationGate, g1_row.gate_id)
    g1_evidence = list(await db.scalars(select(OrchestrationEvidence).where(OrchestrationEvidence.gate_id == g1_gate.id)))
    assert g1_gate.status == "accepted" and [(e.source_type, e.verdict) for e in g1_evidence] == [("child_goal", "accepted")]
    for _ in range(3):
        await service.tick(db, area_run.id)
        probe_row = await _item(db, area_goal, "probe")
        if probe_row is not None:
            break
    probe_task = await db.get(Task, probe_row.task_id)
    await _tick_until(db, service, area_run, lambda: probe_task.status == "in_progress")
    for goal in (root, area_goal):
        await _assert_budget_conserved(db, goal)

    # 9. Safe replan while the probe runs: v2 narrows unreleased g2; released g1/probe stay byte-identical.
    replan = await service.execute_request_roadmap_replan_action(
        db, area_run.id, {"action_type": "request_roadmap_replan", "agent_id": str(planner.id),
                          "scope": "Narrow g2 after g1.", "reason": "g1 findings shrink g2."},
        f"run:{area_run.id}:kind:request_roadmap_replan:version:{v1.version}")
    replan_task = await db.get(Task, replan.target_id)
    await _tick_until(db, service, area_run, lambda: replan_task.status == "in_progress")
    assert await _live_count(db) == 2  # probe + replan planner: the shared cap holds across levels
    g2_narrow = {**g2, "objective": "g2 narrowed", "allocation": {"max_tokens": 100, "max_turns": 4, "max_hours": 1}}
    await _finish(db, project, area_run, replan_task, json.dumps(_roadmap_envelope([g1, probe, g2_narrow], "Narrow g2.")))
    artifact_id = str(plan_artifact_id(replan_task.id, area_run.plan_state["planning_version"]))
    scripts[area_run.id] = lambda run: {"action_type": "accept_plan", "plan_artifact_id": artifact_id} if (
        (run.plan_state or {}).get("pending_replan")) else None
    decision = None
    for _ in range(4):
        await service.tick(db, area_run.id)
        decision = await db.scalar(select(OrchestrationAuthorityDecision).where(
            OrchestrationAuthorityDecision.run_id == area_run.id, OrchestrationAuthorityDecision.status == "pending",
            OrchestrationAuthorityDecision.decision_key.like("roadmap_plan:%")))
        if decision is not None:
            break
    assert decision is not None, area_run.plan_state
    await OrchestrationAuthorityDecisionService().answer_decision(
        db, decision, selected_option="approve", decided_by_user_id=area_goal.manager_user_id)
    roadmap = OrchestrationRoadmapService(service)
    for _ in range(4):
        await service.tick(db, area_run.id)
        v2 = await roadmap.current_version(db, area_goal.id)
        if v2.id != v1.id:
            break
    scripts.pop(area_run.id)
    assert v2.version == v1.version + 1 and not area_run.plan_state.get("pending_replan")
    assert [await _item(db, area_goal, key) for key in ("g1", "probe")] == [g1_row, probe_row]
    assert (g1_row.first_version_id, g1_row.child_goal_id, probe_row.task_id) == (v1.id, leaf.id, probe_task.id)
    assert list(await db.scalars(select(OrchestrationEvidence).where(
        OrchestrationEvidence.gate_id == g1_gate.id))) == g1_evidence and g1_gate.status == "accepted"
    assert await _children_of(db, area_goal) == [leaf] and probe_task.status == "in_progress"

    # 10. The probe completes and is verified; g2 releases from v2's narrowed allocation.
    await _finish(db, project, area_run, probe_task, _report("Probe passed."))
    await _verify(db, project, service, area_run, await db.get(OrchestrationGate, probe_row.gate_id), verifier)
    auto_start["on"] = False  # remaining sessions are played through the shared Roadmap helpers
    for _ in range(3):
        await service.tick(db, area_run.id)
        g2_row = await _item(db, area_goal, "g2")
        if g2_row is not None:
            break
    assert g2_row.first_version_id == v2.id and g2_row.item_snapshot["objective"] == "g2 narrowed"
    reservation = await db.scalar(select(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.child_goal_id == g2_row.child_goal_id))
    assert {k: Decimal(str(v)) for k, v in reservation.allocation.items()} == {
        "max_tokens": 100, "max_turns": 4, "max_hours": 1}
    g2_goal = await db.get(OrchestrationGoal, g2_row.child_goal_id)
    assert (g2_goal.goal_type, g2_goal.goal_type_source) == ("outcome", "parent_explicit")
    assert classified == [ROOT_OBJECTIVE, "g1"]
    for goal in (root, area_goal):
        await _assert_budget_conserved(db, goal)
    # 11. Finite limits are enforced from the root's frozen policy: no deeper level, no fourth descendant.
    assert "child depth 3 exceeds max_goal_depth 2" in await roadmap._expansion_limit_error(db, g2_goal, 1)
    assert "4 descendant goals exceed max_descendant_goals 3" in await roadmap._expansion_limit_error(db, area_goal, 1)

    # 12. g2 runs its own Outcome lifecycle; C then P close only through independent integration.
    await _complete_child(db, project, roadmap, g2_row, planner, producer, verifier, summarizer)
    await _tick_until(db, service, area_run, lambda: g2_row.completed_at is not None)
    models = (OrchestrationAction, Task, OrchestrationGate, OrchestrationEvidence, OrchestrationRoadmapItem,
              OrchestrationRoadmapVersion, OrchestrationBudgetReservation, OrchestrationGoal, OrchestrationRun, Session)
    for _ in range(4):  # C's integration gate opens; then nothing else is due anywhere
        await service.tick(db, area_run.id)
        if await _integration_gate(db, roadmap, area_goal, area_run, v2) is not None:
            break
    await _assert_replay_adds_nothing(db, service, models, (leaf_run, area_run, root_run))
    await _close_roadmap(db, project, service, area_goal, area_run, verifier)
    area_row = await _item(db, root, "area")
    await _tick_until(db, service, root_run, lambda: area_row.completed_at is not None)
    await _close_roadmap(db, project, service, root, root_run, verifier)
    for goal in (root, area_goal):
        await _assert_budget_conserved(db, goal)

    # 13. One row per lineage key: no duplicate tasks, children, artifacts, reservations, or evidence.
    assert len(await _children_of(db, root)) == 1 and len(await _children_of(db, area_goal)) == 2
    assert await _count(db, OrchestrationBudgetReservation) == 3
    assert await _count(db, OrchestrationBudgetReservation, OrchestrationBudgetReservation.status != "settled") == 0
    keys = [a.idempotency_key for a in await db.scalars(select(OrchestrationAction))]
    assert len(keys) == len(set(keys))
    child_evidence = list(await db.scalars(select(OrchestrationEvidence).where(
        OrchestrationEvidence.source_type == "child_goal")))
    item_gates = [(await _item(db, goal, key)).gate_id for goal, key in ((area_goal, "g1"), (area_goal, "g2"), (root, "area"))]
    assert sorted(e.gate_id for e in child_evidence) == sorted(item_gates)
    plan_artifacts = list(await db.scalars(select(Artifact.linked_task_id).where(Artifact.artifact_type == "plan")))
    assert len(plan_artifacts) == len(set(plan_artifacts))
    sessions = [s.task_id for s in await db.scalars(select(Session).where(Session.status != "failed"))]
    assert len(sessions) == len(set(sessions))  # one successful session per task (audit-1 had one failed attempt)
    await _assert_replay_adds_nothing(db, service, models, (leaf_run, area_run, root_run))
    assert external_calls == [] and classified == [ROOT_OBJECTIVE, "g1"]


# --- scenario 2: explicit types are never classified; a corrected Auto root needs a fresh Start ---------

async def test_explicit_type_is_preserved_and_corrected_auto_root_needs_fresh_start(
    db_session, test_project, monkeypatch, external_calls, classifier, auto_start,
):
    db, project = db_session, test_project
    shapes, classified = classifier
    shapes.update({"Pick one thing.": ("outcome", "delivery"), "Plan a big launch.": ("outcome", "delivery")})
    planner = _agent("fix-planner", ["planning", "implementation"])
    db.add(planner)
    await db.flush()
    service = OrchestrationService()
    scripts = {}
    _script(monkeypatch, service, scripts)

    def create(objective, **kw):
        return service.create_goal(db, project.id, OrchestrationGoalCreate(
            objective=objective, success_criteria=[{"key": "done", "description": "Done."}], constraints={},
            budget={"caps": {"max_tokens": 1000, "max_turns": 20, "max_hours": 5}}, **kw,
        ), created_by_user_id=None)

    # Explicit Roadmap: the classifier (which would say Outcome) is never consulted.
    explicit, explicit_run = await create("Plan a big launch.", goal_type="roadmap")
    assert (explicit.goal_type, explicit.goal_type_source, explicit_run.goal_type_snapshot) == (
        "roadmap", "user", "roadmap")
    assert classified == []

    # Auto root resolves Outcome, is Started, and requests a plan on its first run.
    goal, old = await create("Pick one thing.")
    assert (goal.goal_type, goal.goal_type_source, goal.planning_mode) == ("outcome", "auto", "delivery")
    assert classified == ["Pick one thing."]
    old.baseline_authorized = True
    await complete_baseline_processes(db, goal, old)
    await service.tick(db, old.id)
    await service.start_run(db, project.id, goal.id, actor="human:e2e")
    await bind_plan_author_team(db, goal, planner)
    await heal_baseline_drift_for_test(db, goal, old)
    scripts[old.id] = lambda run: None if (run.plan_state or {}).get("planning_task_id") else {
        "action_type": "request_plan", "agent_id": str(planner.id), "work_function": "planning",
        "scope": "Plan it."}
    await service.tick(db, old.id)
    old_plan = await db.get(Task, uuid.UUID(old.plan_state["planning_task_id"]))
    assert old_plan.status == "backlog"  # never activated (auto start off)

    # Correct Auto -> Roadmap after Start: preview, then explicit apply with the preview fingerprint.
    shape = OrchestrationGoalShapeService()
    fields = {"requested_goal_type": "roadmap", "expected_run_id": old.id,
              "expected_contract_fingerprint": compute_goal_contract_fingerprint(goal), "rationale": "It is a roadmap."}
    preview = await shape.preview_correction(db, goal, GoalTypeCorrectionPreviewRequest(**fields), "human:e2e")
    assert preview.requires_new_start and preview.blockers == []
    goal, successor = await shape.apply_correction(db, goal, GoalTypeCorrectionApplyRequest(
        **fields, preview_fingerprint=preview.preview_fingerprint), "human:e2e")
    assert (goal.goal_type, goal.goal_type_source, goal.planning_mode) == ("roadmap", "user", "roadmap")
    assert (old.status, old.terminal_reason, old.goal_type_snapshot) == ("cancelled", "reclassified", "outcome")
    assert (successor.supersedes_run_id, successor.goal_type_snapshot, successor.phase) == (old.id, "roadmap", "baseline")
    assert old_plan.status == "cancelled" and classified == ["Pick one thing."]

    # The old Start does not carry over: the successor plans nothing until its own baseline and Start.
    scripts[successor.id] = lambda run: None if (run.plan_state or {}).get("planning_task_id") else {
        "action_type": "request_plan", "agent_id": str(planner.id), "work_function": "planning",
        "scope": "Plan the Roadmap."}
    await complete_baseline_processes(db, goal, successor)
    await service.tick(db, successor.id)
    assert successor.phase == "ready" and not (successor.plan_state or {}).get("planning_task_id")
    denied = await effective_authorization(db, goal, successor)
    assert (denied.allowed, denied.reason_code) == (False, "start_required"), denied
    old_auth = await effective_authorization(db, goal, old)
    assert not old_auth.allowed
    _, started = await service.start_run(db, project.id, goal.id, actor="human:e2e")
    assert started.id == successor.id and started.phase == "authorized"
    assert successor.goal_contract_fingerprint == compute_goal_contract_fingerprint(goal)
    allowed = await effective_authorization(db, goal, successor)
    assert (allowed.allowed, allowed.authority_source, allowed.run_id) == (True, "root_start", successor.id)
    await bind_plan_author_team(db, goal, planner)
    await heal_baseline_drift_for_test(db, goal, successor)
    await service.tick(db, successor.id)
    plan = await db.get(Task, uuid.UUID(successor.plan_state["planning_task_id"]))
    assert plan.metadata_["orchestration"]["run_id"] == str(successor.id)
    assert plan.metadata_["orchestration_contract"]["report_schema"]["plan_kind"] == "roadmap"
    assert external_calls == []


# --- scenario 3: one failed predecessor attempt, retried, then its dependent releases --------------

async def _approved_retry(db, service, run, task):
    """the failed attempt starts one recovery lineage instead of an automatic retry; this stands in
    for the owner-approved fix, whose retry runs the same executor under a recovery effect key."""
    triggers = [a for a in await _run_actions(db, run.id)
                if a.action_type == "recovery_trigger" and a.request["blocker_code"] == "task_failed"]
    assert len(triggers) == 1
    assert not [a for a in await _run_actions(db, run.id) if a.action_type == "retry_task"]
    return await service.execute_retry_task_action(
        db, run_id=run.id, request={"action_type": "retry_task", "task_id": str(task.id)},
        idempotency_key=f"recovery:{uuid.uuid4()}:retry:1:effect",
    )


async def test_failed_predecessor_attempt_retries_then_its_accepted_gate_releases_the_dependent(
    db_session, test_project, monkeypatch, external_calls, safe_effectiveness_review_continue,
):
    db, project = db_session, test_project
    worker, verifier = _agent("pred-worker", ["implementation"]), _agent("pred-verifier", ["validation"])
    worker.role = "developer"
    db.add_all([worker, verifier])
    await db.flush()
    service, goal, run = await _authorized_run(db, project)
    goal.weight = "trivial"
    _script(monkeypatch, service, {})
    item = {"work_function": "implementation", "scope": "s", "deliverable": "d", "agent_id": str(worker.id)}
    await _seed_accepted_plan(db, project, service, run, worker, plan_items=[
        {**item, "id": "first"}, {**item, "id": "second", "depends_on": ["first"]}])
    await service.tick(db, run.id)

    def released(item_id):
        return [t for t in tasks if t.metadata_["orchestration"].get("plan_item_id") == item_id]

    tasks = await _tasks_for_run(db, run.id)
    (first,) = released("first")
    assert first.status == "in_progress" and released("second") == []
    await _finish(db, project, run, first, None, failed=True)  # the failed predecessor attempt
    await service.tick(db, run.id)
    retry = await _approved_retry(db, service, run, first)
    assert retry.status == "completed" and first.status == "in_progress", retry.error
    tasks = await _tasks_for_run(db, run.id)
    assert released("second") == []  # a failed predecessor is a wait, never a release
    await _finish(db, project, run, first, _report("first delivered on retry"))
    await service.tick(db, run.id)
    await _verify(db, project, service, run, await _item_gate(db, run, "first"), verifier)
    for _ in range(3):
        await service.tick(db, run.id)
    tasks = await _tasks_for_run(db, run.id)
    assert (await _item_gate(db, run, "first")).status == "accepted"
    assert len(released("second")) == 1
    assert external_calls == []


# --- migration checklist: 049 backfills legacy rows with SQL alone ----------------------------------

def test_migration_049_backfill_never_reaches_the_classifier():
    """Upgrade/downgrade behaviour is covered by test_migration_049_goal_planning_hierarchy.py; this pins that
    the backfill is plain SQL (alembic + sqlalchemy only), so no service, classifier, or provider can run."""
    path = Path(__file__).resolve().parent.parent / "alembic" / "versions" / "049_goal_planning_hierarchy.py"
    tree = ast.parse(path.read_text())
    imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                for alias in node.names}
    imported |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert imported == {"alembic", "sqlalchemy"}
