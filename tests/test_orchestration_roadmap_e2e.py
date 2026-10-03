"""Release-gate compositions over real Roadmap runtime events.

Only reusable drivers are imported; each collected case owns its lineage.
"""

import json
import uuid
from copy import deepcopy
from datetime import datetime
from decimal import Decimal

import pytest
import pytest_asyncio
from sqlalchemy import delete, event, func, inspect, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from huddleroom.models.artifact import Artifact
from huddleroom.models.agent import Agent
from huddleroom.models.orchestration import (
    OrchestrationAction, OrchestrationBudgetReservation, OrchestrationEvidence,
    OrchestrationGate, OrchestrationGoal, OrchestrationRoadmapItem,
    OrchestrationRoadmapVersion, OrchestrationRun,
)
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.services.event_bus import emit_event_once
from huddleroom.services.orchestration_budget_service import OrchestrationBudgetService
from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
from huddleroom.services.orchestration_roadmap_service import OrchestrationRoadmapService, parse_roadmap_items
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.schemas.orchestration import OrchestrationGoalCreate
from tests.conftest import complete_baseline_processes, heal_baseline_drift_for_test
from tests.test_orchestration_roadmap_children import goal_item
from tests.test_orchestration_roadmap_integration import (
    _child_actors, _complete_child, _complete_roadmap_task,
    _finish_integration_verification, _integration_gate, _local_decisions,
    _record_measured_completion,
)
from tests.test_orchestration_roadmap_task_items import roadmap_task
from tests.test_orchestration_runtime_e2e import _agent, _run_actions, _seed_accepted_plan, _tasks_for_run


pytestmark = pytest.mark.asyncio


async def _ready_roadmap(db, project):
    service = OrchestrationService()
    # Roster changes after baseline make the real baseline-review reconciler
    # correctly re-open; seed the planner before taking its baseline snapshot.
    planner = _agent("e2e-roadmap-planner", ["planning"])
    db.add(planner)
    await db.flush()
    goal, run = await service.create_goal(
        db, project_id=project.id,
        data=OrchestrationGoalCreate(
            objective="Deliver build, rollout, and docs.",
            success_criteria=[{"key": "delivered", "description": "Roadmap is integrated."}],
            constraints={}, budget={"caps": {"max_tokens": 1000, "max_turns": 10, "max_hours": 10}},
        ), created_by_user_id=None,
    )
    goal.goal_type = "roadmap"
    run.baseline_authorized = True
    await complete_baseline_processes(db, goal, run)
    await service.tick(db, run.id)
    assert (run.phase, run.status) == ("ready", "running")
    await service.start_run(db, project.id, goal.id, actor="human:e2e")
    await service.start_run(db, project.id, goal.id, actor="human:e2e")
    assert (run.phase, run.status) == ("authorized", "running")
    assert await db.scalar(select(func.count()).select_from(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id,
        OrchestrationAction.action_type == "authorize_execution")) == 1
    return service, goal, run, planner


async def _approved_roadmap(db, project, items, *, accept=True):
    """Admit version one through the public planning and authority flow."""
    service, goal, run, planner = await _ready_roadmap(db, project)
    goal.orchestrator_context = {"team": {"agent_ids": [str(agent_id) for agent_id in await db.scalars(
        select(Agent.id).where(Agent.is_active.is_(True))
    )]}}
    request = await service.execute_request_plan_action(
        db, run.id,
        {"action_type": "request_plan", "agent_id": str(planner.id),
         "work_function": "planning", "scope": "Plan the approved Roadmap."},
        f"run:{run.id}:kind:e2e-roadmap-plan",
    )
    artifact = Artifact(project_id=project.id, name="e2e-roadmap-plan", artifact_type="plan", status="draft",
        linked_task_id=request.target_id, created_by_agent=planner.id, metadata_={"plan_items": items})
    db.add(artifact)
    await db.flush()
    normalized = [item.model_dump(mode="json") for item in parse_roadmap_items(items, set(goal.budget["caps"]))]
    fingerprint = service._accepted_plan_fingerprint(normalized)
    decision = await OrchestrationAuthorityDecisionService().create_pending(
        db, goal.id, decision_key=f"roadmap_plan:{fingerprint}", title="Approve Roadmap",
        question="Approve the bounded Roadmap?", authority="human", options=[{"key": "approve"}], run_id=run.id)
    await OrchestrationAuthorityDecisionService().answer_decision(
        db, decision, selected_option="approve", decided_by_user_id=goal.manager_user_id)
    if not accept:
        return service, goal, run, artifact
    await service.execute_accept_plan_action(
        db, run.id, {"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
        f"run:{run.id}:kind:e2e-roadmap-accept")
    version = await OrchestrationRoadmapService(service).current_version(db, goal.id)
    assert version is not None
    goal.weight = "trivial"  # Keep the unrelated effectiveness analyzer deterministic.
    return service, goal, run, version


async def _item(db, goal, key):
    return await db.scalar(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == goal.id,
        OrchestrationRoadmapItem.item_key == key))


async def _snapshot(db, models):
    """Compare durable column values, including JSON and all lineage IDs."""
    await db.flush()
    def value(row, key):
        result = getattr(row, key)
        return result.replace(tzinfo=None) if isinstance(result, datetime) else deepcopy(result)

    return {
        model.__name__: [
            {column.key: value(row, column.key) for column in inspect(model).column_attrs}
            for row in await db.scalars(select(model).order_by(model.id))
        ] for model in models
    }


async def _assert_task_lineage(db, run, *, metadata_key, metadata_value, gate, delegation_key=None):
    tasks = list(await db.scalars(select(Task).where(
        Task.metadata_["orchestration"]["run_id"].as_string() == str(run.id),
        Task.metadata_["orchestration"][metadata_key].as_string() == metadata_value,
    )))
    assert len(tasks) == 1, [(task.id, task.metadata_) for task in tasks]
    task = tasks[0]
    if delegation_key:
        actions = list(await db.scalars(select(OrchestrationAction).where(
            OrchestrationAction.run_id == run.id,
            OrchestrationAction.idempotency_key == delegation_key,
            OrchestrationAction.action_type == "create_delegation_task",
        )))
        assert len(actions) == 1 and actions[0].target_id == task.id
    evidence = list(await db.scalars(select(OrchestrationEvidence).where(
        OrchestrationEvidence.run_id == run.id, OrchestrationEvidence.gate_id == gate.id,
    )))
    report = [row for row in evidence if row.source_type == "task" and row.source_id == task.id]
    assert len(report) == 1 and report[0].verdict == "accepted", [(e.source_type, e.verdict) for e in evidence]
    verifier_actions = list(await db.scalars(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == "request_verification",
        OrchestrationAction.request["gate_id"].as_string() == str(gate.id),
    )))
    assert len(verifier_actions) == 1
    verifier_tasks = list(await db.scalars(select(Task).where(
        Task.metadata_["orchestration"]["run_id"].as_string() == str(run.id),
        Task.metadata_["orchestration"]["verification_gate_id"].as_string() == str(gate.id),
    )))
    assert len(verifier_tasks) == 1 and verifier_tasks[0].id == verifier_actions[0].target_id
    proof = [row for row in evidence if row.source_type == "verification"]
    assert len(proof) == 1 and proof[0].verdict == "accepted"
    assert proof[0].evidence_metadata["task_id"] == str(verifier_tasks[0].id)


async def _assert_plan_and_summary(db, run):
    tasks = list(await db.scalars(select(Task).where(
        Task.metadata_["orchestration"]["run_id"].as_string() == str(run.id))))
    planning = [task for task in tasks if "orchestration_plan" in task.metadata_]
    summaries = [task for task in tasks if task.metadata_["orchestration"].get("final_summary")]
    assert len(planning) == len(summaries) == 1
    assert str(planning[0].id) == run.plan_state["planning_task_id"]
    for kind, source, target in (
        ("plan_accepted", "artifact", run.plan_state["accepted_artifact_id"]),
        ("final_summary_accepted", "session", None),
    ):
        gates = list(await db.scalars(select(OrchestrationGate).where(
            OrchestrationGate.run_id == run.id, OrchestrationGate.gate_type == kind)))
        assert len(gates) == 1 and gates[0].status == "accepted"
        evidence = list(await db.scalars(select(OrchestrationEvidence).where(
            OrchestrationEvidence.run_id == run.id, OrchestrationEvidence.gate_id == gates[0].id,
            OrchestrationEvidence.source_type == source)))
        assert len(evidence) == 1 and evidence[0].verdict == "accepted"
        if target:
            assert str(evidence[0].source_id) == target
        else:
            assert evidence[0].evidence_metadata["task_id"] == str(summaries[0].id)
    summary_delegations = list(await db.scalars(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id,
        OrchestrationAction.idempotency_key == f"run:{run.id}:kind:create_delegation_task:final_summary")))
    assert len(summary_delegations) == 1 and summary_delegations[0].target_id == summaries[0].id


async def _active_child_work(db, project, service, row, planner, producer):
    child_run = await db.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == row.child_goal_id))
    await _seed_accepted_plan(db, project, service, child_run, planner, [{
        "id": "child-work", "work_function": "implementation", "scope": "Deliver the child outcome.",
        "deliverable": "Verified child work", "agent_id": str(producer.id), "success_criterion_keys": ["local"],
    }])
    assert (await service.tick(db, child_run.id))["authorized_execution"]["step"] == "release_work"
    task = next(task for task in await _tasks_for_run(db, child_run.id)
                if task.metadata_["orchestration"].get("plan_item_id") == "child-work")
    session = await _active_session(db, project.id, task)
    return child_run, task, session


async def _active_session(db, project_id, task):
    task.status = "in_progress"
    session = Session(project_id=project_id, task_id=task.id, agent_id=task.assigned_to,
                      adapter_type="test", status="running", metadata_={})
    db.add(session)
    await db.flush()
    return session


async def _late_output(db, project_id, task, session):
    # Model an adapter callback arriving after the cancellation transaction.
    # The callback supplies an output event; it cannot un-cancel the owned task.
    session.output = json.dumps({"status": "done", "verdict": "accepted", "evidence": ["late report"]})
    await db.flush()
    await emit_event_once(db, project_id, "session.completed",
                          {"session_id": str(session.id), "task_id": str(task.id)},
                          dedup_key=f"task8-late-output:{session.id}")


async def _assert_budget_conserved(db, parent):
    summary = await OrchestrationBudgetService().remaining(db, parent)
    for key, cap in summary["caps"].items():
        assert Decimal(cap) == sum(Decimal(summary[field][key]) for field in (
            "direct_spend", "settled_child_spend", "active_reservations", "active_commitments", "remaining"))
    return summary


async def test_roadmap_golden_path_reaches_independent_integration_closeout(
    db_session, test_project, monkeypatch, safe_effectiveness_review_continue,
):
    """Baseline-ready -> Start -> build/child/docs -> integration -> closeout."""
    planner, child_producer, verifier, child_summarizer = await _child_actors(db_session)
    builder = _agent("e2e-build-docs", ["implementation"])
    parent_summarizer = _agent("e2e-parent-summary", ["summarization"])
    db_session.add_all([builder, parent_summarizer])
    await db_session.flush()
    build = {**roadmap_task("build"), "agent_id": str(builder.id)}
    rollout = goal_item("rollout", allocation={"max_tokens": 400, "max_turns": 5, "max_hours": 1})
    rollout["depends_on"] = ["build"]
    docs = {**roadmap_task("docs", depends_on=["rollout"]), "agent_id": str(builder.id)}
    service, parent, run, version = await _approved_roadmap(db_session, test_project, [build, rollout, docs])
    roadmap = OrchestrationRoadmapService(service)
    parent.weight = "trivial"
    await heal_baseline_drift_for_test(db_session, parent, run)
    _local_decisions(monkeypatch, service, lambda _ctx: {"action_type": "noop", "reason": "Events drive E2E."})

    assert (await service.tick(db_session, run.id))["authorized_execution"]["item_key"] == "build"
    build_row = await _item(db_session, parent, "build")
    released = await _complete_roadmap_task(db_session, test_project.id, service, run, build_row, verifier.id)
    assert (await db_session.get(OrchestrationGate, build_row.gate_id)).status == "accepted"
    assert released[1]["authorized_execution"]["item_key"] == "rollout"
    rollout_row = await _item(db_session, parent, "rollout")
    child, child_run, child_gate, child_evidence, _ = await _complete_child(
        db_session, test_project, OrchestrationRoadmapService(service), rollout_row,
        planner, child_producer, verifier, child_summarizer)
    assert child_gate.status == "accepted"
    assert service._criterion_evidence_manifest([child_gate], child_evidence, parent, str(version.id)) == {}
    docs_row = None
    for _ in range(3):
        await service.tick(db_session, run.id)
        docs_row = await _item(db_session, parent, "docs")
        if docs_row is not None:
            break
    assert docs_row is not None, "accepted child gate must settle/project before docs releases"
    reservation = await db_session.scalar(select(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.roadmap_item_id == rollout_row.id))
    assert reservation.status == "settled" and rollout_row.completed_at is not None
    await _complete_roadmap_task(db_session, test_project.id, service, run, docs_row, verifier.id)
    integration = await _integration_gate(db_session, OrchestrationRoadmapService(service), parent, run, version)
    assert integration.status == "open"
    await service.execute_request_verification_action(
        db_session, run.id,
        {"action_type": "request_verification", "gate_id": str(integration.id), "work_function": "validation"},
        f"run:{run.id}:kind:request_verification:integration:{version.id}")
    await _finish_integration_verification(db_session, test_project.id, service, run, integration, verifier.id)
    await service.tick(db_session, run.id)
    proof = await db_session.scalar(select(OrchestrationEvidence).where(
        OrchestrationEvidence.gate_id == integration.id,
        OrchestrationEvidence.source_type == "verification",
        OrchestrationEvidence.verdict == "accepted"))
    assert proof is not None
    summary = next(action for action in await _run_actions(db_session, run.id)
                   if action.action_type == "request_final_summary")
    summary_task = await db_session.get(__import__("huddleroom.models.task", fromlist=["Task"]).Task, summary.target_id)
    await _record_measured_completion(db_session, test_project.id, run, summary_task, summary_task.assigned_to, json.dumps({
        "summary": "Independent integration accepted the Roadmap.",
        "criteria": [
            {"criterion_key": key, "evidence_ids": [str(proof.id)]}
            for key in service._declared_success_criterion_keys(parent)
        ], "unresolved_gaps": [],
    }))
    for _ in range(6):
        await service.tick(db_session, run.id)
        if run.status == "completed":
            break
    await db_session.refresh(parent)
    assert (parent.status, run.status, run.phase) == ("completed", "completed", "completed"), [
        (gate.gate_type, gate.status, gate.failure_reason)
        for gate in await db_session.scalars(select(OrchestrationGate).where(OrchestrationGate.run_id == run.id))
    ]

    for _ in range(2):
        await service.tick(db_session, run.id)
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationRoadmapVersion).where(
        OrchestrationRoadmapVersion.goal_id == parent.id)) == 1
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == parent.id)) == 3
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationGoal).where(
        OrchestrationGoal.parent_goal_id == parent.id)) == 1
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.parent_goal_id == parent.id)) == 1
    for action_type in ("authorize_execution", "request_plan", "accept_plan", "request_final_summary", "complete_run"):
        assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
            OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == action_type)) == 1
    assert child.status == "completed" and child_run.status == "completed"
    # Every release key has one durable action, target, gate and accepted proof;
    # replay above is deliberately after the complete closeout sequence.
    for key, row, source_type in (
        ("build", build_row, "verification"),
        ("rollout", rollout_row, "child_goal"),
        ("docs", docs_row, "verification"),
    ):
        release = await db_session.scalar(select(OrchestrationAction).where(
            OrchestrationAction.run_id == run.id,
            OrchestrationAction.idempotency_key == f"run:{run.id}:kind:release_roadmap_item:{key}"))
        assert release is not None and release.status == "completed"
        assert release.target_id == (row.task_id or row.child_goal_id)
        assert await db_session.scalar(select(func.count()).select_from(OrchestrationGate).where(
            OrchestrationGate.id == row.gate_id)) == 1
        assert await db_session.scalar(select(func.count()).select_from(OrchestrationEvidence).where(
            OrchestrationEvidence.gate_id == row.gate_id,
            OrchestrationEvidence.source_type == source_type,
            OrchestrationEvidence.verdict == "accepted")) == 1
        if row.task_id:
            assert await db_session.scalar(select(func.count()).select_from(Task).where(Task.id == row.task_id)) == 1
            verification = await db_session.scalar(select(OrchestrationAction).where(
                OrchestrationAction.run_id == run.id,
                OrchestrationAction.action_type == "request_verification",
                OrchestrationAction.request["gate_id"].as_string() == str(row.gate_id)))
            assert verification is not None and verification.target_id is not None
            assert await db_session.scalar(select(func.count()).select_from(Task).where(Task.id == verification.target_id)) == 1

    child_actions = await _run_actions(db_session, child_run.id)
    for action_type in ("authorize_execution", "request_plan", "accept_plan", "request_final_summary", "complete_run"):
        assert len([action for action in child_actions if action.action_type == action_type]) == 1
    child_work_actions = [action for action in child_actions if action.action_type == "request_verification"]
    assert len(child_work_actions) == 1 and child_work_actions[0].target_id is not None
    assert await db_session.scalar(select(func.count()).select_from(Task).where(
        Task.id == child_work_actions[0].target_id)) == 1
    integration_action = await db_session.scalar(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id,
        OrchestrationAction.action_type == "request_verification",
        OrchestrationAction.request["gate_id"].as_string() == str(integration.id)))
    assert integration_action is not None and integration_action.target_id is not None
    assert await db_session.scalar(select(func.count()).select_from(Task).where(Task.id == integration_action.target_id)) == 1
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationGate).where(
        OrchestrationGate.id == integration.id)) == 1
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationEvidence).where(
        OrchestrationEvidence.gate_id == integration.id,
        OrchestrationEvidence.source_type == "verification",
        OrchestrationEvidence.verdict == "accepted")) == 1

    for current_run in (run, child_run):
        await _assert_plan_and_summary(db_session, current_run)
    for row in (build_row, docs_row):
        gates = list(await db_session.scalars(select(OrchestrationGate).where(
            OrchestrationGate.run_id == run.id,
            OrchestrationGate.required_evidence["roadmap_item_key"].as_string() == row.item_key)))
        assert len(gates) == 1 and gates[0].id == row.gate_id
        await _assert_task_lineage(
            db_session, run, metadata_key="roadmap_item_key", metadata_value=row.item_key, gate=gates[0],
            delegation_key=f"run:{run.id}:kind:create_delegation_task:roadmap_item:{row.item_key}")
    await _assert_task_lineage(
        db_session, child_run, metadata_key="plan_item_id", metadata_value="child-work", gate=child_gate,
        delegation_key=f"run:{child_run.id}:kind:create_delegation_task:plan_item:child-work")
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationGate).where(
        OrchestrationGate.run_id == child_run.id,
        OrchestrationGate.required_evidence["plan_item_id"].as_string() == "child-work")) == 1
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
        OrchestrationAction.run_id == child_run.id,
        OrchestrationAction.idempotency_key == f"run:{child_run.id}:kind:expand_plan_item:plan_item:child-work")) == 1
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.run_id == run.id,
        OrchestrationAuthorityDecision.decision_key == f"roadmap_plan:{version.fingerprint}",
        OrchestrationAuthorityDecision.status == "answered",
        OrchestrationAuthorityDecision.selected_option == "approve")) == 1
    assert await db_session.scalar(select(func.count()).select_from(Task).where(
        Task.metadata_["orchestration"]["run_id"].as_string() == str(run.id),
        Task.metadata_["orchestration"]["verification_gate_id"].as_string() == str(integration.id))) == 1
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == "request_verification",
        OrchestrationAction.request["gate_id"].as_string() == str(integration.id))) == 1
    # Replaying terminal events/ticks must preserve every durable row, not merely
    # whichever primary key happened to be returned by the original action.
    models = (OrchestrationAction, Task, OrchestrationGate, OrchestrationEvidence,
              OrchestrationRoadmapItem, OrchestrationRoadmapVersion, OrchestrationBudgetReservation)
    before = await _snapshot(db_session, models)
    await service.tick(db_session, child_run.id)
    await service.tick(db_session, run.id)
    after = await _snapshot(db_session, models)
    for model, rows in before.items():
        assert [row["id"] for row in after[model]] == [row["id"] for row in rows], model


@pytest.mark.parametrize("boundary", ["version_insert", "reservation_check", "child_insert", "child_evidence_projection", "settlement", "integration_gate_creation"])
async def test_roadmap_fault_boundaries_rollback_then_replay_one_lineage(
    db_session, test_project, test_engine, monkeypatch, boundary,
    safe_effectiveness_review_continue, _committed_lineage_cleanup,
):
    """Six different SQL operations fail, roll back, then recover in a new session."""
    actors = await _child_actors(db_session)
    service, parent, run, artifact_or_version = await _approved_roadmap(
        db_session, test_project,
        [goal_item("child", allocation={"max_tokens": 400, "max_turns": 5, "max_hours": 1})],
        accept=boundary != "version_insert")
    parent.weight = "trivial"
    await heal_baseline_drift_for_test(db_session, parent, run)
    _local_decisions(monkeypatch, service, lambda _ctx: (
        {"action_type": "accept_plan", "plan_artifact_id": str(artifact_or_version.id)}
        if boundary == "version_insert" else {"action_type": "noop", "reason": "Recovery check"}))
    if boundary in {"child_evidence_projection", "settlement", "integration_gate_creation"}:
        await service.tick(db_session, run.id)
        row = await _item(db_session, parent, "child")
        await _complete_child(db_session, test_project, OrchestrationRoadmapService(service), row, *actors)
        if boundary == "integration_gate_creation":
            assert (await service.tick(db_session, run.id))["authorized_execution"] == {
                "step": "settle_children", "count": 1}
    run_id, parent_id = run.id, parent.id
    models = (OrchestrationGoal, OrchestrationRun, OrchestrationAction, OrchestrationRoadmapVersion,
              OrchestrationRoadmapItem, OrchestrationBudgetReservation, OrchestrationGate, OrchestrationEvidence)
    before = await _snapshot(db_session, models)
    await db_session.commit()
    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    model, hook, predicate = {
        "version_insert": (OrchestrationRoadmapVersion, "after_insert", lambda row: True),
        # reserve_child has checked the remaining budget before its INSERT flush.
        "reservation_check": (OrchestrationBudgetReservation, "before_insert", lambda row: True),
        "child_insert": (OrchestrationGoal, "after_insert", lambda row: row.parent_goal_id == parent_id),
        "child_evidence_projection": (OrchestrationEvidence, "after_insert", lambda row: row.source_type == "child_goal"),
        "settlement": (OrchestrationBudgetReservation, "after_update", lambda row: row.status == "settled"),
        "integration_gate_creation": (OrchestrationGate, "after_insert", lambda row: row.gate_type == "roadmap_integration"),
    }[boundary]
    hits = []

    def fail_at_operation(_mapper, _connection, row):
        if predicate(row):
            hits.append(row.id)
            raise RuntimeError(f"{boundary}: injected")

    event.listen(model, hook, fail_at_operation)
    try:
        async with factory() as failed:
            with pytest.raises(RuntimeError, match=f"{boundary}: injected"):
                async with failed.begin():
                    result = await service.tick(failed, run_id)
                    assert hits, result
            await failed.rollback()
    finally:
        event.remove(model, hook, fail_at_operation)
    assert len(hits) == 1
    async with factory() as restarted:
        assert await _snapshot(restarted, models) == before
        recovered = await service.tick(restarted, run_id)
        assert recovered["authorized_execution"]["step"] == {
            "version_insert": "plan_decision", "reservation_check": "release_item", "child_insert": "release_item",
            "child_evidence_projection": "settle_children", "settlement": "settle_children",
            "integration_gate_creation": "integration_decision",
        }[boundary]
        await restarted.commit()
    async with factory() as replay:
        parent = await replay.get(OrchestrationGoal, parent_id)
        if boundary == "version_insert":
            _local_decisions(monkeypatch, service, lambda _ctx: {"action_type": "noop", "reason": "Replay"})
        # Version admission may release its first child on the next tick;
        # settlement may create integration. Freeze after those expected steps.
        await service.tick(replay, run_id)
        await service.tick(replay, run_id)
        action_ids = [action.id for action in await _run_actions(replay, run_id)]
        stable = await _snapshot(replay, (OrchestrationRoadmapVersion, OrchestrationRoadmapItem,
                                        OrchestrationBudgetReservation, OrchestrationGate, OrchestrationEvidence))
        await service.tick(replay, run_id)
        assert [action.id for action in await _run_actions(replay, run_id)] == action_ids
        assert await _snapshot(replay, (OrchestrationRoadmapVersion, OrchestrationRoadmapItem,
                                       OrchestrationBudgetReservation, OrchestrationGate, OrchestrationEvidence)) == stable
        versions = list(await replay.scalars(select(OrchestrationRoadmapVersion).where(
            OrchestrationRoadmapVersion.goal_id == parent_id)))
        items = list(await replay.scalars(select(OrchestrationRoadmapItem).where(
            OrchestrationRoadmapItem.goal_id == parent_id)))
        children = list(await replay.scalars(select(OrchestrationGoal).where(
            OrchestrationGoal.parent_goal_id == parent_id)))
        reservations = list(await replay.scalars(select(OrchestrationBudgetReservation).where(
            OrchestrationBudgetReservation.parent_goal_id == parent_id)))
        assert len(versions) == len(items) == len(children) == len(reservations) == 1
        assert items[0].child_goal_id == children[0].id == reservations[0].child_goal_id
        assert reservations[0].roadmap_item_id == items[0].id
        releases = list(await replay.scalars(select(OrchestrationAction).where(
            OrchestrationAction.run_id == run_id, OrchestrationAction.action_type == "release_roadmap_item")))
        assert len(releases) == 1 and releases[0].status == "completed" and releases[0].target_id == children[0].id
        for action_type in ("authorize_execution", "request_plan", "accept_plan"):
            assert await replay.scalar(select(func.count()).select_from(OrchestrationAction).where(
                OrchestrationAction.run_id == run_id, OrchestrationAction.action_type == action_type)) == 1
        child_runs = list(await replay.scalars(select(OrchestrationRun).where(
            OrchestrationRun.goal_id == children[0].id)))
        assert len(child_runs) == 1
        assert await replay.scalar(select(func.count()).select_from(OrchestrationAction).where(
            OrchestrationAction.run_id == child_runs[0].id,
            OrchestrationAction.action_type == "authorize_execution")) == 1
        gates = list(await replay.scalars(select(OrchestrationGate).where(
            OrchestrationGate.run_id == run_id, OrchestrationGate.gate_type == "child_goal_completed")))
        assert len(gates) == 1 and gates[0].id == items[0].gate_id
        projected = list(await replay.scalars(select(OrchestrationEvidence).where(
            OrchestrationEvidence.run_id == run_id, OrchestrationEvidence.source_type == "child_goal")))
        integration = list(await replay.scalars(select(OrchestrationGate).where(
            OrchestrationGate.run_id == run_id, OrchestrationGate.gate_type == "roadmap_integration")))
        completed = boundary in {"child_evidence_projection", "settlement", "integration_gate_creation"}
        assert len(projected) == len(integration) == int(completed)
        assert reservations[0].status == ("settled" if completed else "active")
        if completed:
            assert projected[0].source_id == children[0].id and projected[0].gate_id == gates[0].id
            assert projected[0].verdict == gates[0].status == "accepted"


@pytest_asyncio.fixture
async def _committed_lineage_cleanup(test_engine, db_session):
    yield
    await db_session.rollback()
    # Immutable child/version FKs require child lineages to be removed before
    # conftest's ordinary per-table cleanup after these committed-session tests.
    async with test_engine.begin() as connection:
        await connection.execute(delete(OrchestrationBudgetReservation))
        await connection.execute(delete(OrchestrationRoadmapItem))
        await connection.execute(delete(OrchestrationGoal).where(OrchestrationGoal.parent_goal_id.is_not(None)))


@pytest.mark.parametrize("failure,reason", [
    ("cycle", "plan_revision_required"),
    ("missing_staging", "waiting_unstaged_approval"),
    ("rejected_approval", "needs_attention"),
    ("budget_exhaustion", "budget_wait"),
    ("missing_telemetry", "needs_attention"),
    ("child_cancellation", "needs_attention"),
    ("late_output", "needs_attention"),
])
async def test_roadmap_failures_wait_without_scope_or_budget_expansion(
    db_session, test_project, monkeypatch, safe_effectiveness_review_continue, failure, reason,
):
    actors = await _child_actors(db_session)
    first = goal_item("child", allocation={"max_tokens": 400, "max_turns": 5, "max_hours": 1})
    successor = {**goal_item("successor", allocation={"max_tokens": 400, "max_turns": 4, "max_hours": 1}),
                 "depends_on": ["child"]}
    if failure in {"missing_staging", "rejected_approval"}:
        first["mutates_shared_state"] = True
    service, parent, run, version = await _approved_roadmap(
        db_session, test_project, [first, successor], accept=failure != "cycle")
    parent.weight = "trivial"
    await heal_baseline_drift_for_test(db_session, parent, run)
    decision = {"action_type": "noop", "reason": "Wait matrix"}
    _local_decisions(monkeypatch, service, lambda _ctx: decision)
    late = None
    if failure == "cycle":
        first["depends_on"] = ["successor"]
        version.metadata_ = {"plan_items": [first, successor]}
        decision = {"action_type": "accept_plan", "plan_artifact_id": str(version.id)}
    elif failure == "rejected_approval":
        assert (await service.tick(db_session, run.id))["authorized_execution"] == {
            "step": "waiting", "reason": "waiting_unstaged_approval"}
        pending = list(await db_session.scalars(select(OrchestrationAuthorityDecision).where(
            OrchestrationAuthorityDecision.run_id == run.id,
            OrchestrationAuthorityDecision.status == "pending")))
        assert len(pending) == 1 and pending[0].authority == "human"
        await OrchestrationAuthorityDecisionService().answer_decision(
            db_session, pending[0], selected_option="reject", decided_by_user_id=parent.manager_user_id)
    elif failure == "budget_exhaustion":
        task = await db_session.get(Task, uuid.UUID(run.plan_state["planning_task_id"]))
        measured = await _record_measured_completion(
            db_session, test_project.id, run, task, task.assigned_to, json.dumps({"status": "done"}))
        measured.metadata_ = {"token_count_in": 650, "token_count_out": 0}
    elif failure in {"missing_telemetry", "child_cancellation", "late_output"}:
        assert (await service.tick(db_session, run.id))["authorized_execution"]["item_key"] == "child"
        row = await _item(db_session, parent, "child")
        if failure == "missing_telemetry":
            _, child_run, _, _, sessions = await _complete_child(
                db_session, test_project, OrchestrationRoadmapService(service), row, *actors)
            sessions[0].metadata_ = {}
        else:
            child_run, child_task, child_session = await _active_child_work(
                db_session, test_project, service, row, actors[0], actors[1])
            await service.cancel_goal(db_session, test_project.id, row.child_goal_id)
            if failure == "late_output":
                late = child_run, child_task, child_session
    await db_session.flush()
    contract = deepcopy((parent.objective, parent.success_criteria, parent.constraints, parent.budget))
    versions = await _snapshot(db_session, (OrchestrationRoadmapVersion,))
    budget_before = await _assert_budget_conserved(db_session, parent)
    if late:
        child_run, child_task, child_session = late
        child_before = await _snapshot(db_session, (OrchestrationEvidence, OrchestrationGate))
        await _late_output(db_session, test_project.id, child_task, child_session)
        await service.tick(db_session, child_run.id)
        assert await _snapshot(db_session, (OrchestrationEvidence, OrchestrationGate)) == child_before
    wait_actions = None
    for _ in range(2):
        result = (await service.tick(db_session, run.id))["authorized_execution"]
        if failure == "cycle":
            assert result["step"] == reason
            assert run.plan_state["revision_reason"] == "Roadmap dependency cycle detected"
        else:
            assert result == {"step": "waiting", "reason": reason}
        assert (parent.objective, parent.success_criteria, parent.constraints, parent.budget) == contract
        assert await _snapshot(db_session, (OrchestrationRoadmapVersion,)) == versions
        budget_after = await _assert_budget_conserved(db_session, parent)
        assert budget_after["remaining"] == budget_before["remaining"]
        assert await _item(db_session, parent, "successor") is None
        actions = [(action.id, action.idempotency_key, action.status) for action in await _run_actions(db_session, run.id)]
        if wait_actions is not None:
            assert actions == wait_actions
        wait_actions = actions
    await db_session.refresh(run)
    blockers = [item for item in run.active_blockers if item.get("kind") in {
        "staging_boundary", "budget_measurement", "child_cancelled"}]
    expected_kind = {"rejected_approval": "staging_boundary", "missing_telemetry": "budget_measurement",
                     "child_cancellation": "child_cancelled", "late_output": "child_cancelled"}.get(failure)
    assert [item["kind"] for item in blockers] == ([expected_kind] if expected_kind else [])
    if expected_kind:
        assert service.run_condition(parent, run) == "needs_attention"
    if failure in {"missing_staging", "rejected_approval"}:
        decisions = list(await db_session.scalars(select(OrchestrationAuthorityDecision).where(
            OrchestrationAuthorityDecision.run_id == run.id,
            OrchestrationAuthorityDecision.decision_key == f"roadmap_unstaged_mutation:{version.id}:child")))
        assert len(decisions) == 1 and decisions[0].authority == "human"
        assert decisions[0].status == ("pending" if failure == "missing_staging" else "answered")
    rows = list(await db_session.scalars(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == parent.id)))
    assert len(rows) == int(failure in {"missing_telemetry", "child_cancellation", "late_output"})
    if rows:
        assert (await db_session.get(OrchestrationGate, rows[0].gate_id)).status != "accepted"
        assert await db_session.scalar(select(func.count()).select_from(OrchestrationEvidence).where(
            OrchestrationEvidence.gate_id == rows[0].gate_id)) == 0
        reservation = (await db_session.scalars(select(OrchestrationBudgetReservation).where(
            OrchestrationBudgetReservation.roadmap_item_id == rows[0].id))).one()
        assert reservation.status == ("active" if failure == "missing_telemetry" else "settled")
        if reservation.status == "settled":
            assert reservation.settled_spend == reservation.allocation
            settled = await _snapshot(db_session, (OrchestrationBudgetReservation,))
            await service.tick(db_session, run.id)
            assert await _snapshot(db_session, (OrchestrationBudgetReservation,)) == settled
