import json
import uuid
from datetime import timedelta
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import func, select

from huddleroom.config import settings
from huddleroom.models.orchestration import (
    OrchestrationAction, OrchestrationBudgetReservation, OrchestrationEvidence, OrchestrationGate,
    OrchestrationGoal, OrchestrationRoadmapItem, OrchestrationRoadmapVersion, OrchestrationRun,
)
from huddleroom.models.artifact import Artifact
from huddleroom.models.agent import Agent
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
from huddleroom.services.session_sync import sync_task_from_session
from tests.orchestration_wake_helpers import NOOP_WAKE_WHEN
from tests.test_orchestration_roadmap_children import goal_item
from tests.test_orchestration_roadmap_task_items import roadmap_task
from tests.test_orchestration_runtime_e2e import (
    _agent, _authorized_run, _complete_task_session, _run_actions, _seed_accepted_plan, _tasks_for_run,
)
from huddleroom.services.orchestration_roadmap_service import OrchestrationRoadmapService, parse_roadmap_items
from huddleroom.services.orchestration_service import OrchestrationService


@pytest.fixture(autouse=True)
def _manual_task_start(monkeypatch):
    # These tests drive the release-then-manual-run flow; auto-start is covered in test_orchestration_task_autostart.py.
    async def _noop(self, db, goal, run):
        return 0
    monkeypatch.setattr(OrchestrationService, "_start_released_tasks", _noop)


pytestmark = pytest.mark.asyncio


async def accepted_goal_roadmap(db, project, item, cap=1000, team_agent_ids=(), budget_caps=None):
    """Accept a mixed Roadmap via the public plan action, preserving plan_state."""
    planner = _agent("mixed-roadmap-planner", ["planning"])
    db.add(planner)
    await db.flush()
    service, parent, run = await _authorized_run(db, project)
    parent.goal_type = "roadmap"
    parent.budget = {"caps": budget_caps or {"max_tokens": cap, "max_turns": 10, "max_hours": 10}}
    items = item if isinstance(item, list) else [item]
    request = await service.execute_request_plan_action(
        db, run.id,
        {"action_type": "request_plan", "agent_id": str(planner.id), "work_function": "planning", "scope": "Plan Roadmap work."},
        f"run:{run.id}:kind:mixed-roadmap-request",
    )
    parent.orchestrator_context = {"team": {"agent_ids": [str(agent_id) for agent_id in (
        team_agent_ids or await db.scalars(select(Agent.id).where(Agent.is_active.is_(True)))
    )]}}
    artifact = Artifact(
        project_id=project.id, name="mixed-roadmap-plan", artifact_type="plan", status="draft",
        linked_task_id=request.target_id, created_by_agent=planner.id, metadata_={"plan_items": items},
    )
    db.add(artifact)
    await db.flush()
    normalized = [item.model_dump(mode="json") for item in parse_roadmap_items(items, set(parent.budget["caps"]))]
    fingerprint = service._accepted_plan_fingerprint(normalized)
    decision = await OrchestrationAuthorityDecisionService().create_pending(
        db, parent.id, decision_key=f"roadmap_plan:{fingerprint}", title="Approve Roadmap plan",
        question="Approve?", authority="human", options=[{"key": "approve"}], run_id=run.id,
    )
    await OrchestrationAuthorityDecisionService().answer_decision(
        db, decision, selected_option="approve", decided_by_user_id=parent.manager_user_id,
    )
    await service.execute_accept_plan_action(
        db, run.id, {"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
        f"run:{run.id}:kind:mixed-roadmap-accept",
    )
    roadmap = OrchestrationRoadmapService(service)
    version = await roadmap.current_version(db, parent.id)
    assert run.plan_state["status"] == "accepted" and version is not None
    return roadmap, parent, run, version


async def release(db, project, item=None, cap=1000):
    roadmap, parent, run, version = await accepted_goal_roadmap(db, project, item or child_item(), cap)
    result = await roadmap.orchestration.tick(db, run.id)
    row = await db.scalar(select(OrchestrationRoadmapItem).where(OrchestrationRoadmapItem.goal_id == parent.id))
    return roadmap, parent, run, result, row


def child_item(key="child", **kwargs):
    allocation = kwargs.pop("allocation", {"max_tokens": 400, "max_turns": 5, "max_hours": 1})
    return goal_item(
        key, allocation=allocation, **kwargs,
    )


async def _record_measured_completion(db, project_id, run, task, agent_id, output):
    session = await _complete_task_session(db, project_id, task, agent_id, output)
    session.metadata_ = {**session.metadata_, "token_count_in": 1, "token_count_out": 1}
    session.started_at = run.started_at
    session.ended_at = run.started_at + timedelta(minutes=1)
    await sync_task_from_session(db, session)
    await db.flush()
    return session


async def _complete_roadmap_task(db, project_id, service, run, row, verifier_id, *, complete_producer=True):
    task = await db.get(Task, row.task_id)
    producer = None
    if complete_producer:
        producer = await _record_measured_completion(
            db, project_id, run, task, task.assigned_to, json.dumps({"status": "done"}),
        )
    verification = await service.execute_request_verification_action(
        db, run.id,
        {"action_type": "request_verification", "gate_id": str(row.gate_id), "work_function": "validation"},
        f"run:{run.id}:kind:request_verification:roadmap_item:{row.item_key}",
    )
    verifier_task = await db.get(Task, verification.target_id)
    assert verifier_task.assigned_to == verifier_id
    assert (verification.status, verification.target_type, verification.target_id) == ("completed", "task", verifier_task.id)
    assert await service._outcome_verification_action_for_task(db, run, await db.get(OrchestrationGate, row.gate_id), verifier_task)
    await _record_measured_completion(
        db, project_id, run, verifier_task, verifier_id,
        json.dumps({"status": "done", "verdict": "accepted", "evidence": ["independent check"]}),
    )
    result = await service.tick(db, run.id)
    gate = await db.get(OrchestrationGate, row.gate_id)
    assert gate.status == "accepted", {
        "failure": gate.failure_reason,
        "request": {key: verification.request.get(key) for key in ("source_task_id", "producer_agent_id", "verifier_agent_id")},
        "binding": {key: verifier_task.metadata_["orchestration"].get(key) for key in ("verification_gate_id", "source_task_id", "producer_agent_id", "verifier_agent_id")},
        "evidence": [(item.source_type, item.verdict) for item in await db.scalars(
            select(OrchestrationEvidence).where(OrchestrationEvidence.gate_id == row.gate_id)
        )],
    }
    return producer, result


async def test_integration_decision_commits_authority_wait(db_session, test_project, monkeypatch):
    worker = _agent("integration-worker", ["implementation"])
    verifier = _agent("integration-verifier", ["validation"])
    db_session.add_all([worker, verifier])
    await db_session.flush()
    item = roadmap_task("integration")
    item["agent_id"] = str(worker.id)
    roadmap, parent, run, _, row = await release(db_session, test_project, item)
    service = roadmap.orchestration
    task = await db_session.get(Task, row.task_id)
    await _record_measured_completion(
        db_session, test_project.id, run, task, task.assigned_to, json.dumps({"status": "done"}),
    )
    verification = await service.execute_request_verification_action(
        db_session, run.id,
        {"action_type": "request_verification", "gate_id": str(row.gate_id), "work_function": "validation"},
        f"run:{run.id}:kind:request_verification:roadmap_item:{row.item_key}",
    )
    verifier_task = await db_session.get(Task, verification.target_id)
    await _record_measured_completion(
        db_session, test_project.id, run, verifier_task, verifier_task.assigned_to,
        json.dumps({"status": "done", "verdict": "accepted", "evidence": ["verified"]}),
    )

    async def decision(*_args, **_kwargs):
        return SimpleNamespace(parsed_decision={})

    async def wait_for_authority(*_args, **_kwargs):
        raise HTTPException(status_code=409, detail="waiting_human_scope_budget_authority")

    monkeypatch.setattr(service, "request_llm_decision", decision)
    monkeypatch.setattr(service, "_dispatch_execution_decision", wait_for_authority)
    result = await service.tick(db_session, run.id)
    assert result["authorized_execution"] == {
        "step": "waiting", "reason": "waiting_human_scope_budget_authority",
    }


async def _complete_child(db, project, roadmap, row, planner, producer, verifier, summarizer):
    """Drive the normal Outcome lifecycle; no terminal state or evidence is seeded."""
    service = roadmap.orchestration
    child = await db.get(OrchestrationGoal, row.child_goal_id)
    child_run = await db.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == child.id))
    child.weight = "trivial"  # Fixture-only: avoid an unrelated human closeout decision.
    await _seed_accepted_plan(
        db, project, service, child_run, planner,
        [{
            "id": "child-work", "work_function": "implementation", "scope": "Deliver the child outcome.",
            "deliverable": "Independently verified child work.", "agent_id": str(producer.id),
            "success_criterion_keys": ["local"],
        }],
    )
    assert (await service.tick(db, child_run.id))["authorized_execution"]["step"] == "release_work"
    work_task = next(task for task in await _tasks_for_run(db, child_run.id)
                     if service._task_work_function(task) == "implementation")
    work_gate = await db.get(OrchestrationGate, uuid.UUID(work_task.metadata_["orchestration"]["plan_item_gate_id"]))
    producer_session = await _record_measured_completion(
        db, project.id, child_run, work_task, producer.id,
        json.dumps({"status": "done", "changes": ["child delivered"]}),
    )
    verification = await service.execute_request_verification_action(
        db, child_run.id,
        {"action_type": "request_verification", "gate_id": str(work_gate.id), "work_function": "validation"},
        f"run:{child_run.id}:kind:request_verification:child-work",
    )
    verifier_task = await db.get(Task, verification.target_id)
    assert verifier_task.assigned_to == verifier.id
    verifier_session = await _record_measured_completion(
        db, project.id, child_run, verifier_task, verifier.id,
        json.dumps({"status": "done", "verdict": "accepted", "evidence": ["independent child check"]}),
    )
    # A fully accepted child has no further deterministic work after ingestion;
    # suppress the unrelated next-action LLM branch while retaining tick ingestion.
    original_advance = service._advance_authorized_execution

    async def no_next_action(*_args, **_kwargs):
        return {"step": "waiting"}

    service._advance_authorized_execution = no_next_action
    await service.tick(db, child_run.id)
    service._advance_authorized_execution = original_advance
    await db.refresh(work_gate)
    assert work_gate.status == "accepted"
    accepted_evidence = await db.scalar(select(OrchestrationEvidence.id).where(
        OrchestrationEvidence.gate_id == work_gate.id,
        OrchestrationEvidence.source_type == "verification",
        OrchestrationEvidence.verdict == "accepted",
    ))
    summary = next(action for action in await _run_actions(db, child_run.id)
                   if action.action_type == "request_final_summary")
    summary_task = await db.get(Task, summary.target_id)
    assert summary_task.assigned_to == summarizer.id
    summary_session = await _record_measured_completion(
        db, project.id, child_run, summary_task, summarizer.id,
        json.dumps({
            "summary": "Child evidence is complete.",
            "criteria": [{"criterion_key": "local", "evidence_ids": [str(accepted_evidence)]}],
            "unresolved_gaps": [],
        }),
    )
    service._advance_authorized_execution = no_next_action
    await service.tick(db, child_run.id)
    service._advance_authorized_execution = original_advance
    await db.refresh(child)
    await db.refresh(child_run)
    assert (child.status, child_run.status, child_run.phase) == ("completed", "completed", "completed")
    evidence = list((await db.scalars(select(OrchestrationEvidence).where(
        OrchestrationEvidence.gate_id == work_gate.id,
        OrchestrationEvidence.verdict == "accepted",
    ))).all())
    return child, child_run, work_gate, evidence, (producer_session, verifier_session, summary_session)


async def _child_actors(db):
    actors = (
        _agent("child-planner", ["planning"]), _agent("child-producer", ["implementation"]),
        _agent("child-verifier", ["validation"]), _agent("child-summarizer", ["summarization"]),
    )
    db.add_all(actors)
    await db.flush()
    return actors


async def test_mixed_task_goal_task_graph_releases_in_gate_order(
    db_session, test_project, safe_effectiveness_review_continue,
):
    planner, producer, verifier, summarizer = await _child_actors(db_session)
    build_agent = _agent("mixed-builder", ["implementation"])
    db_session.add(build_agent)
    await db_session.flush()
    build = {**roadmap_task("build"), "agent_id": str(build_agent.id)}
    rollout = {**child_item("rollout"), "depends_on": ["build"]}
    docs = {**roadmap_task("docs", depends_on=["rollout"]), "agent_id": str(build_agent.id)}
    roadmap, parent, run, _ = await accepted_goal_roadmap(
        db_session, test_project, [build, rollout, docs],
        team_agent_ids=[planner.id, producer.id, verifier.id, summarizer.id, build_agent.id],
    )
    service = roadmap.orchestration

    assert (await service.tick(db_session, run.id))["authorized_execution"]["item_key"] == "build"
    build_row = await db_session.scalar(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == parent.id, OrchestrationRoadmapItem.item_key == "build",
    ))
    _producer_session, released = await _complete_roadmap_task(
        db_session, test_project.id, service, run, build_row, verifier.id,
    )
    assert released["authorized_execution"] == {"step": "release_item", "item_key": "rollout", "unit_type": "goal"}
    rollout_row = await db_session.scalar(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == parent.id, OrchestrationRoadmapItem.item_key == "rollout",
    ))
    await _complete_child(db_session, test_project, roadmap, rollout_row, planner, producer, verifier, summarizer)
    assert (await service.tick(db_session, run.id))["authorized_execution"] == {"step": "settle_children", "count": 1}
    assert (await service.tick(db_session, run.id))["authorized_execution"] == {
        "step": "release_item", "item_key": "docs", "unit_type": "task",
    }


async def test_mixed_cap_releases_eligible_goal_after_two_active_tasks(db_session, test_project):
    builder = _agent("cap-builder", ["implementation"])
    db_session.add(builder)
    await db_session.flush()
    roadmap, parent, run, _ = await accepted_goal_roadmap(db_session, test_project, [
        {**roadmap_task("one", mutates_shared_state=False), "agent_id": str(builder.id)},
        {**roadmap_task("two", mutates_shared_state=False), "agent_id": str(builder.id)},
        child_item("child"),
    ])
    service = roadmap.orchestration

    assert (await service.tick(db_session, run.id))["authorized_execution"]["item_key"] == "one"
    assert (await service.tick(db_session, run.id))["authorized_execution"]["item_key"] == "two"
    assert (await service.tick(db_session, run.id))["authorized_execution"] == {
        "step": "release_item", "item_key": "child", "unit_type": "goal",
    }
    rows = list((await db_session.scalars(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == parent.id,
    ).order_by(OrchestrationRoadmapItem.item_key))).all())
    task_rows = [row for row in rows if row.unit_type == "task"]
    child_row, = [row for row in rows if row.unit_type == "goal"]
    tasks = [await db_session.get(Task, row.task_id) for row in task_rows]
    assert [(row.item_key, row.unit_type) for row in rows] == [
        ("child", "goal"), ("one", "task"), ("two", "task"),
    ]
    assert all(task.status in {"backlog", "ready", "in_progress", "blocked"} for task in tasks)
    assert await db_session.get(OrchestrationGoal, child_row.child_goal_id) is not None

    async def counts():
        lineages = list((await db_session.scalars(select(OrchestrationRoadmapItem).where(
            OrchestrationRoadmapItem.goal_id == parent.id,
        ))).all())
        return (
            len(lineages), len([row for row in lineages if row.unit_type == "task"]),
            await db_session.scalar(select(func.count()).select_from(OrchestrationGoal).where(
                OrchestrationGoal.parent_goal_id == parent.id,
            )),
            await db_session.scalar(select(func.count()).select_from(OrchestrationRun).join(
                OrchestrationGoal, OrchestrationRun.goal_id == OrchestrationGoal.id,
            ).where(OrchestrationGoal.parent_goal_id == parent.id)),
            len([action for action in await _run_actions(db_session, run.id)
                 if action.action_type == "release_roadmap_item"]),
        )

    assert await counts() == (3, 2, 1, 1, 3)

    assert (await service.tick(db_session, run.id))["authorized_execution"] == {
        "step": "waiting", "reason": "waiting_active_work",
    }
    assert await counts() == (3, 2, 1, 1, 3)


async def test_completed_child_projects_one_parent_item_evidence_record(
    db_session, test_project, safe_effectiveness_review_continue,
):
    actors = await _child_actors(db_session)
    roadmap, parent, run, _, row = await release(db_session, test_project, child_item())
    child, child_run, child_gate, child_evidence, _ = await _complete_child(
        db_session, test_project, roadmap, row, *actors,
    )

    assert (await roadmap.orchestration.tick(db_session, run.id))["authorized_execution"] == {
        "step": "settle_children", "count": 1,
    }
    evidence = list((await db_session.scalars(select(OrchestrationEvidence).where(
        OrchestrationEvidence.gate_id == row.gate_id,
    ))).all())
    assert [(item.source_type, item.source_id, item.verdict) for item in evidence] == [("child_goal", child.id, "accepted")]
    assert evidence[0].id == uuid.uuid5(uuid.NAMESPACE_URL, f"rally:child-rollup:{run.id}:{row.item_key}:{child.id}")
    manifest_gates, manifest_evidence = await roadmap.orchestration._accepted_non_summary_manifest(
        db_session, child_run.id,
    )
    assert child_gate.id in [item.id for item in manifest_gates]
    assert set(item.id for item in child_evidence).issubset(item.id for item in manifest_evidence)
    assert evidence[0].evidence_metadata == {
        "child_run_id": str(child_run.id),
        "child_gate_ids": [str(item.id) for item in manifest_gates],
        "child_evidence_ids": [str(item.id) for item in manifest_evidence],
    }
    parent_gates = list((await db_session.scalars(select(OrchestrationGate).where(
        OrchestrationGate.run_id == run.id, OrchestrationGate.success_criterion_key == "done",
    ))).all())
    assert not parent_gates or all(gate.status != "accepted" for gate in parent_gates)


async def test_terminal_child_without_completed_outcome_does_not_unlock_parent(db_session, test_project):
    planner, producer, _verifier, _summarizer = await _child_actors(db_session)
    roadmap, parent, run, _, row = await release(db_session, test_project, child_item())
    child_run = await db_session.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == row.child_goal_id))
    await _seed_accepted_plan(
        db_session, test_project, roadmap.orchestration, child_run, planner,
        [{"id": "work", "work_function": "implementation", "scope": "Do child work.",
          "deliverable": "Child work.", "agent_id": str(producer.id), "success_criterion_keys": ["local"]}],
    )
    await roadmap.orchestration.tick(db_session, child_run.id)
    task = next(task for task in await _tasks_for_run(db_session, child_run.id)
                if roadmap.orchestration._task_work_function(task) == "implementation")
    await _record_measured_completion(
        db_session, test_project.id, child_run, task, producer.id,
        json.dumps({"status": "done", "changes": ["terminal task only"]}),
    )
    child = await db_session.get(OrchestrationGoal, row.child_goal_id)
    assert child.status == "active"
    assert (await roadmap.orchestration.tick(db_session, run.id))["authorized_execution"] == {
        "step": "waiting", "reason": "waiting_active_work",
    }
    assert list((await db_session.scalars(select(OrchestrationEvidence).where(
        OrchestrationEvidence.gate_id == row.gate_id,
    ))).all()) == []


@pytest.mark.parametrize("field", ["parent_goal_id", "roadmap_version_id", "roadmap_item_key"])
async def test_wrong_child_version_or_item_cannot_roll_up(
    db_session, test_project, safe_effectiveness_review_continue, field,
):
    actors = await _child_actors(db_session)
    roadmap, parent, run, _, row = await release(db_session, test_project, child_item())
    child, _child_run, _child_gate, _child_evidence, _ = await _complete_child(
        db_session, test_project, roadmap, row, *actors,
    )
    if field == "parent_goal_id":
        other_parent = OrchestrationGoal(
            project_id=test_project.id, objective="other parent", original_request="other parent",
            success_criteria=[], constraints={}, budget={}, orchestrator_context={},
        )
        db_session.add(other_parent)
        await db_session.flush()
        child.parent_goal_id = other_parent.id
    elif field == "roadmap_version_id":
        from huddleroom.models.orchestration import OrchestrationRoadmapVersion

        current = await roadmap.current_version(db_session, parent.id)
        bad_version = OrchestrationRoadmapVersion(
            goal_id=parent.id, run_id=run.id, version=current.version + 1,
            plan_artifact_id=current.plan_artifact_id, snapshot=current.snapshot,
            fingerprint=f"bad-{uuid.uuid4().hex}", approval_reference=current.approval_reference,
        )
        db_session.add(bad_version)
        await db_session.flush()
        child.roadmap_version_id = bad_version.id
    else:
        child.roadmap_item_key = "wrong"
    await db_session.flush()

    with db_session.no_autoflush:
        with pytest.raises(HTTPException, match="lineage is invalid") as exc:
            await roadmap.orchestration.tick(db_session, run.id)
    assert exc.value.status_code == 409
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationEvidence).where(
        OrchestrationEvidence.gate_id == row.gate_id,
    )) == 0


async def test_replay_settles_and_projects_child_once(
    db_session, test_project, safe_effectiveness_review_continue, monkeypatch,
):
    actors = await _child_actors(db_session)
    roadmap, parent, run, _, row = await release(db_session, test_project, child_item())
    child, _child_run, _child_gate, _child_evidence, _ = await _complete_child(
        db_session, test_project, roadmap, row, *actors,
    )
    assert (await roadmap.orchestration.tick(db_session, run.id))["authorized_execution"] == {
        "step": "settle_children", "count": 1,
    }
    _local_decisions(monkeypatch, roadmap.orchestration, lambda _ctx: {
        "action_type": "noop", "reason": "Awaiting integration verification.",
    })
    assert (await roadmap.orchestration.tick(db_session, run.id))["authorized_execution"]["step"] == "integration_decision"
    reservations = list((await db_session.scalars(select(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.roadmap_item_id == row.id,
    ))).all())
    assert [(item.id, item.status, item.settlement_reason) for item in reservations] == [(
        uuid.uuid5(uuid.NAMESPACE_URL, f"rally:budget-reservation:{parent.id}:{row.item_key}"), "settled", "completed",
    )]
    evidence = list((await db_session.scalars(select(OrchestrationEvidence).where(
        OrchestrationEvidence.gate_id == row.gate_id,
    ))).all())
    assert [item.id for item in evidence] == [
        uuid.uuid5(uuid.NAMESPACE_URL, f"rally:child-rollup:{run.id}:{row.item_key}:{child.id}"),
    ]


async def test_terminal_child_settles_before_parent_measurement_attention_and_replays_after_repair(
    db_session, test_project, safe_effectiveness_review_continue, monkeypatch,
):
    actors = await _child_actors(db_session)
    roadmap, parent, run, _, row = await release(db_session, test_project, child_item())
    parent_plan = next(action for action in await _run_actions(db_session, run.id)
                       if action.action_type == "request_plan")
    parent_task = await db_session.get(Task, parent_plan.target_id)
    parent_session = await _complete_task_session(
        db_session, test_project.id, parent_task, parent_task.assigned_to,
        json.dumps({"status": "done"}),
    )
    parent_session.started_at = run.started_at
    parent_session.ended_at = run.started_at + timedelta(minutes=1)
    parent_session.metadata_ = {}
    await db_session.flush()

    child, _child_run, _child_gate, _child_evidence, _ = await _complete_child(
        db_session, test_project, roadmap, row, *actors,
    )
    result = await roadmap.orchestration.tick(db_session, run.id)
    reservation = await db_session.scalar(select(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.roadmap_item_id == row.id,
    ))
    evidence = list((await db_session.scalars(select(OrchestrationEvidence).where(
        OrchestrationEvidence.gate_id == row.gate_id,
    ))).all())
    assert result["authorized_execution"] == {"step": "waiting", "reason": "needs_attention"}
    assert reservation.status == "settled"
    assert row.completed_at is not None
    assert [(item.source_type, item.source_id, item.verdict) for item in evidence] == [
        ("child_goal", child.id, "accepted"),
    ]
    assert any(item.get("scope") == f"parent:{parent.id}" for item in run.active_blockers)

    parent_session.metadata_ = {"token_count_in": 1, "token_count_out": 1}
    await db_session.flush()
    _local_decisions(monkeypatch, roadmap.orchestration, lambda _ctx: {
        "action_type": "noop", "reason": "Awaiting integration verification.",
    })
    result = await roadmap.orchestration.tick(db_session, run.id)
    assert result["authorized_execution"]["step"] == "integration_decision"
    assert not any(item.get("scope") == f"parent:{parent.id}" for item in run.active_blockers)
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationEvidence).where(
        OrchestrationEvidence.gate_id == row.gate_id,
    )) == 1


async def test_no_eligible_item_returns_explicit_wait_reason(db_session, test_project):
    builder = _agent("wait-builder", ["implementation"])
    verifier = _agent("wait-verifier", ["validation"])
    db_session.add_all([builder, verifier])
    await db_session.flush()
    dependency, parent, run, _ = await accepted_goal_roadmap(db_session, test_project, [
        {**roadmap_task("build", mutates_shared_state=False), "agent_id": str(builder.id)},
        {**roadmap_task("after", depends_on=["build"], mutates_shared_state=False), "agent_id": str(builder.id)},
    ])
    assert (await dependency.orchestration.tick(db_session, run.id))["authorized_execution"]["item_key"] == "build"
    assert (await dependency.orchestration.tick(db_session, run.id))["authorized_execution"] == {
        "step": "waiting", "reason": "waiting_dependencies",
    }
    shared, _parent, shared_run, _ = await accepted_goal_roadmap(db_session, test_project, [
        {**roadmap_task("one", mutates_shared_state=True), "agent_id": str(builder.id)},
        {**roadmap_task("two", mutates_shared_state=True), "agent_id": str(builder.id)},
    ])
    await shared.orchestration.tick(db_session, shared_run.id)
    decision = await db_session.scalar(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.run_id == shared_run.id,
        OrchestrationAuthorityDecision.decision_key.like("roadmap_unstaged_mutation:%:one"),
    ))
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, decision, selected_option="approve", decided_by_user_id=_parent.manager_user_id,
    )
    await shared.orchestration.tick(db_session, shared_run.id)
    assert (await shared.orchestration.tick(db_session, shared_run.id))["authorized_execution"] == {
        "step": "waiting", "reason": "waiting_shared_workspace",
    }
    capped, _parent, capped_run, _ = await accepted_goal_roadmap(db_session, test_project, [
        {**roadmap_task(key, mutates_shared_state=False), "agent_id": str(builder.id)} for key in ("one", "two", "three")
    ])
    await capped.orchestration.tick(db_session, capped_run.id)
    await capped.orchestration.tick(db_session, capped_run.id)
    assert (await capped.orchestration.tick(db_session, capped_run.id))["authorized_execution"] == {
        "step": "waiting", "reason": "waiting_active_work",
    }
    unstaged_item = roadmap_task("unstaged", mutates_shared_state=True)
    unstaged_item["staging_boundary"] = None
    unstaged, unstaged_parent, unstaged_run, _ = await accepted_goal_roadmap(
        db_session, test_project, [{**unstaged_item, "agent_id": str(builder.id)}],
    )
    assert (await unstaged.orchestration.tick(db_session, unstaged_run.id))["authorized_execution"] == {
        "step": "waiting", "reason": "waiting_unstaged_approval",
    }
    decision = await OrchestrationAuthorityDecisionService().list_decisions(
        db_session, unstaged_parent.id, status="pending",
    )
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, decision[0], selected_option="reject", decided_by_user_id=unstaged_parent.manager_user_id,
    )
    assert (await unstaged.orchestration.tick(db_session, unstaged_run.id))["authorized_execution"] == {
        "step": "waiting", "reason": "needs_attention",
    }
    over_budget, _parent, over_budget_run, _ = await accepted_goal_roadmap(
        db_session, test_project, [child_item("over-budget", allocation={"max_tokens": 700, "max_turns": 1, "max_hours": 1})],
    )
    spend_task = Task(project_id=test_project.id, title="prior spend", status="done", assigned_to=builder.id,
                      metadata_={"orchestration": {"run_id": str(over_budget_run.id)}})
    db_session.add(spend_task)
    await db_session.flush()
    db_session.add(OrchestrationAction(
        run_id=over_budget_run.id, idempotency_key=f"test:prior-spend:{over_budget_run.id}:{spend_task.id}",
        action_type="create_delegation_task", request={}, target_type="task", target_id=spend_task.id,
        status="completed",
    ))
    await db_session.flush()
    db_session.add(Session(project_id=test_project.id, task_id=spend_task.id, agent_id=builder.id,
                           adapter_type="test", status="completed", started_at=over_budget_run.started_at,
                           ended_at=over_budget_run.started_at + timedelta(minutes=1),
                           metadata_={"token_count_in": 400, "token_count_out": 0}))
    await db_session.flush()
    assert (await over_budget.orchestration.tick(db_session, over_budget_run.id))["authorized_execution"] == {
        "step": "waiting", "reason": "budget_wait",
    }


async def test_missing_telemetry_does_not_stall_later_child_cancellation(
    db_session, test_project, safe_effectiveness_review_continue,
):
    actors = await _child_actors(db_session)
    roadmap, parent, run, _ = await accepted_goal_roadmap(
        db_session, test_project, [child_item("first"), child_item("second")],
    )
    await roadmap.orchestration.tick(db_session, run.id)
    await roadmap.orchestration.tick(db_session, run.id)
    rows = {row.item_key: row for row in (await db_session.scalars(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == parent.id,
    ))).all()}
    _child, _run, _gate, _evidence, sessions = await _complete_child(
        db_session, test_project, roadmap, rows["first"], *actors,
    )
    sessions[0].metadata_ = {}
    child_run = await db_session.scalar(select(OrchestrationRun).where(
        OrchestrationRun.goal_id == rows["second"].child_goal_id,
    ))
    cancelling_task = Task(
        project_id=test_project.id, title="cancelled child work", status="in_progress", assigned_to=actors[1].id,
        metadata_={"orchestration": {"run_id": str(child_run.id)}},
    )
    db_session.add(cancelling_task)
    await db_session.flush()
    db_session.add(OrchestrationAction(
        run_id=child_run.id, idempotency_key=f"test:cancelled-child:{child_run.id}:{cancelling_task.id}",
        action_type="create_delegation_task", request={}, target_type="task", target_id=cancelling_task.id,
        status="completed",
    ))
    await db_session.flush()
    db_session.add(Session(
        project_id=test_project.id, task_id=cancelling_task.id, agent_id=actors[1].id,
        adapter_type="test", status="running", metadata_={},
    ))
    await db_session.flush()
    await roadmap.orchestration.cancel_goal(
        db_session, test_project.id, rows["second"].child_goal_id, cancelled_by="human:test",
    )
    assert (await roadmap.orchestration.tick(db_session, run.id))["authorized_execution"] == {
        "step": "waiting", "reason": "needs_attention",
    }
    second_reservation = await db_session.scalar(select(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.roadmap_item_id == rows["second"].id,
    ))
    assert second_reservation.status == "settled" and second_reservation.measurement_complete is False
    assert any(item["kind"] == "budget_measurement" and item["item_key"] == "first" for item in run.active_blockers)
    assert any(item["kind"] == "child_cancelled" and item["item_key"] == "second" for item in run.active_blockers)


async def test_measured_overage_creates_one_durable_blocker_on_replay(
    db_session, test_project, safe_effectiveness_review_continue,
):
    actors = await _child_actors(db_session)
    roadmap, parent, run, _, row = await release(db_session, test_project, child_item())
    _child, _child_run, _child_gate, _child_evidence, sessions = await _complete_child(
        db_session, test_project, roadmap, row, *actors,
    )
    sessions[0].metadata_ = {"token_count_in": 401, "token_count_out": 1}
    await db_session.flush()
    for _ in range(2):
        assert (await roadmap.orchestration.tick(db_session, run.id))["authorized_execution"] == {
            "step": "waiting", "reason": "needs_attention",
        }
    reservation = await db_session.scalar(select(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.roadmap_item_id == row.id,
    ))
    await db_session.refresh(run)
    await db_session.refresh(row)
    assert reservation.status == "active" and row.completed_at is None
    blockers = [item for item in run.active_blockers if item["kind"] == "budget_integrity"]
    assert len(blockers) == 1 and blockers[0]["item_key"] == row.item_key, {
        "blockers": run.active_blockers, "reservation": reservation.settled_spend,
    }
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationEvidence).where(
        OrchestrationEvidence.gate_id == row.gate_id,
    )) == 0


async def _finish_integration_verification(db, project_id, service, run, gate, verifier_id):
    """Use the dispatched verifier task and its terminal event, never seeded evidence."""
    action = await db.scalar(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id,
        OrchestrationAction.action_type == "request_verification",
        OrchestrationAction.request["gate_id"].as_string() == str(gate.id),
    ).order_by(OrchestrationAction.created_at.desc()))
    assert action is not None and action.status == "completed"
    task = await db.get(Task, action.target_id)
    assert task is not None and task.assigned_to == verifier_id
    await _record_measured_completion(
        db, project_id, run, task, verifier_id,
        json.dumps({"status": "done", "verdict": "accepted", "evidence": ["independent integration check"]}),
    )
    return action, task


async def _integration_gate(db, roadmap, parent, run, version):
    return await db.get(
        OrchestrationGate,
        uuid.uuid5(uuid.NAMESPACE_URL, f"rally:roadmap-integration:{run.id}:{version.id}"),
    )


def _local_decisions(monkeypatch, service, decision_fn):
    """Keep the production dispatcher in the test transaction's SQLite database."""
    async def request(db, run_id, adapter=None):
        run = await db.get(OrchestrationRun, run_id)
        goal = await db.get(OrchestrationGoal, run.goal_id)
        context = await service._decision_context(db, goal, run)
        return await service.record_validated_decision(db, run_id, context, {}, decision_fn(context))

    monkeypatch.setattr(service, "request_llm_decision", request)


async def test_integration_gate_waits_for_every_current_item_gate(db_session, test_project, monkeypatch):
    builder = _agent("integration-builder", ["implementation"])
    verifier = _agent("integration-first-verifier", ["validation"])
    db_session.add_all([builder, verifier])
    await db_session.flush()
    roadmap, parent, run, version = await accepted_goal_roadmap(db_session, test_project, [
        {**roadmap_task("one", mutates_shared_state=False), "agent_id": str(builder.id)},
        {**roadmap_task("two", mutates_shared_state=False), "agent_id": str(builder.id)},
    ])
    service = roadmap.orchestration
    decision = {"action_type": "noop", "reason": "Awaiting item verification."}
    _local_decisions(monkeypatch, service, lambda _ctx: decision)
    assert (await service.tick(db_session, run.id))["authorized_execution"]["item_key"] == "one"
    one = await db_session.scalar(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == parent.id, OrchestrationRoadmapItem.item_key == "one",
    ))
    await _complete_roadmap_task(db_session, test_project.id, service, run, one, verifier.id)
    two = await db_session.scalar(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == parent.id, OrchestrationRoadmapItem.item_key == "two",
    ))
    task = await db_session.get(Task, two.task_id)
    await _record_measured_completion(db_session, test_project.id, run, task, builder.id, json.dumps({"status": "done"}))
    await service.tick(db_session, run.id)
    assert await _integration_gate(db_session, roadmap, parent, run, version) is None

    await _complete_roadmap_task(db_session, test_project.id, service, run, two, verifier.id, complete_producer=False)
    gate = await _integration_gate(db_session, roadmap, parent, run, version)
    assert gate is not None and gate.status == "open" and gate.gate_type == "roadmap_integration"


@pytest.mark.parametrize("accept_v1", [False, True], ids=["open-v1", "accepted-v1"])
async def test_integration_gate_is_created_once_across_replay(
    db_session, test_project, monkeypatch, safe_effectiveness_review_continue, accept_v1,
):
    builder = _agent("integration-replay-builder", ["implementation"])
    verifier = _agent("integration-replay-verifier", ["validation"])
    db_session.add_all([builder, verifier])
    await db_session.flush()
    roadmap, parent, run, version = await accepted_goal_roadmap(db_session, test_project, [
        {**roadmap_task("only", mutates_shared_state=False), "agent_id": str(builder.id)},
    ])
    service = roadmap.orchestration
    parent.weight = "trivial"
    summarizer = _agent("integration-replay-summarizer", ["summarization"])
    db_session.add(summarizer)
    decision = {
        "action_type": "noop", "reason": "Awaiting integration verification.",
    }
    _local_decisions(monkeypatch, service, lambda _ctx: decision)
    await roadmap.orchestration.tick(db_session, run.id)
    row = await db_session.scalar(select(OrchestrationRoadmapItem).where(OrchestrationRoadmapItem.goal_id == parent.id))
    await _complete_roadmap_task(db_session, test_project.id, roadmap.orchestration, run, row, verifier.id)
    first = await _integration_gate(db_session, roadmap, parent, run, version)
    await roadmap.orchestration.tick(db_session, run.id)
    second = await _integration_gate(db_session, roadmap, parent, run, version)
    assert first.id == second.id == uuid.uuid5(
        uuid.NAMESPACE_URL, f"rally:roadmap-integration:{run.id}:{version.id}"
    )
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationGate).where(
        OrchestrationGate.id == first.id
    )) == 1

    if accept_v1:
        decision = {"action_type": "request_verification", "gate_id": str(first.id), "work_function": "validation"}
        await service.tick(db_session, run.id)
        await _finish_integration_verification(db_session, test_project.id, service, run, first, verifier.id)
        decision = {"action_type": "noop", "reason": "Replan before summarizing V1."}

    # Task 7 owns public replan admission.  This immutable V2 setup isolates
    # Task 6's required retained-lineage/current-version integration behavior.
    next_item = {**roadmap_task("next", mutates_shared_state=False), "agent_id": str(builder.id)}
    v2 = OrchestrationRoadmapVersion(
        goal_id=parent.id, run_id=run.id, version=version.version + 1,
        plan_artifact_id=version.plan_artifact_id,
        snapshot={"schema_version": 1, "items": [*version.snapshot["items"], next_item]},
        fingerprint=f"retained-v2-{uuid.uuid4().hex}", approval_reference=version.approval_reference,
    )
    db_session.add(v2)
    await db_session.flush()
    assert (await roadmap.orchestration.tick(db_session, run.id))["authorized_execution"]["item_key"] == "next"
    next_row = await db_session.scalar(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == parent.id, OrchestrationRoadmapItem.item_key == "next",
    ))
    await _complete_roadmap_task(db_session, test_project.id, roadmap.orchestration, run, next_row, verifier.id)
    current = await _integration_gate(db_session, roadmap, parent, run, v2)
    assert current is not None and current.id == uuid.uuid5(
        uuid.NAMESPACE_URL, f"rally:roadmap-integration:{run.id}:{v2.id}",
    )
    assert first.status == ("accepted" if accept_v1 else "open")
    assert not await service._run_ready_for_final_summary_request(db_session, parent, run)
    assert not [action for action in await _run_actions(db_session, run.id)
                if action.action_type in {"request_final_summary", "complete_run"}]

    decision = {"action_type": "request_verification", "gate_id": str(current.id), "work_function": "validation"}
    await service.tick(db_session, run.id)
    await _finish_integration_verification(db_session, test_project.id, service, run, current, verifier.id)
    decision = {"action_type": "noop", "reason": "Current integration verified."}
    await service.tick(db_session, run.id)
    gates, evidence = await service._accepted_non_summary_manifest(db_session, run.id)
    assert first.id not in {gate.id for gate in gates}
    assert {row.gate_id, next_row.gate_id, current.id}.issubset(gate.id for gate in gates)
    assert first.id not in {item.gate_id for item in evidence}
    proof = next(item for item in evidence if item.gate_id == current.id and item.source_type == "verification")
    summary_action = next(action for action in await _run_actions(db_session, run.id)
                          if action.action_type == "request_final_summary")
    summary_task = await db_session.get(Task, summary_action.target_id)
    payload = {
        "summary": "V2 integration verified with retained V1 lineage.",
        "criteria": [{"criterion_key": key, "evidence_ids": [str(proof.id)]}
                     for key in service._declared_success_criterion_keys(parent)],
        "unresolved_gaps": [],
    }
    if accept_v1:
        stale = await db_session.scalar(select(OrchestrationEvidence).where(
            OrchestrationEvidence.gate_id == first.id,
            OrchestrationEvidence.source_type == "verification",
            OrchestrationEvidence.verdict == "accepted",
        ))
        assert stale is not None
        stale_payload = {**payload, "criteria": [
            {**criterion, "evidence_ids": [str(stale.id)]} for criterion in payload["criteria"]
        ]}
        summary_session = await _record_measured_completion(
            db_session, test_project.id, run, summary_task, summarizer.id, json.dumps(stale_payload),
        )
        await service._ingest_evidence_from_events(
            db_session, run, await service._new_events(db_session, test_project.id, run.event_cursor),
        )
        summary_gate = await db_session.get(
            OrchestrationGate, uuid.UUID(summary_task.metadata_["orchestration"]["gate_id"]),
        )
        validated, failure = await service._validated_final_summary_payload(
            db_session, summary_gate, await service._evidence_for_gate(db_session, summary_gate),
        )
        assert validated is None and "accepted verification" in failure
        assert summary_gate.status != "accepted" and run.status != "completed"
        # Correct the report before ticking its real candidate evidence to acceptance.
        summary_session.output = json.dumps(payload)
        await db_session.flush()
    else:
        await _record_measured_completion(
            db_session, test_project.id, run, summary_task, summarizer.id, json.dumps(payload),
        )
    await service.tick(db_session, run.id)
    await service.tick(db_session, run.id)
    assert (parent.status, run.status, run.phase) == ("completed", "completed", "completed")
    closeout = await service._closeout_preconditions_manifest(db_session, parent, run)
    assert first.id not in {uuid.UUID(item["gate_id"]) for item in closeout["accepted_non_summary_gates"]}
    assert closeout["criterion_evidence"] == payload["criteria"]
    actions = await _run_actions(db_session, run.id)
    assert len([action for action in actions if action.action_type == "request_final_summary"]) == 1
    assert len([action for action in actions if action.action_type == "complete_run"]) == 1


async def test_integration_verifier_must_differ_from_every_item_producer(
    db_session, test_project, monkeypatch, safe_effectiveness_review_continue,
):
    builders = (_agent("integration-producer-one", ["implementation"]), _agent("integration-producer-two", ["implementation"]))
    verifier = _agent("integration-independent-verifier", ["validation"])
    db_session.add_all([*builders, verifier])
    await db_session.flush()
    manager = _agent("integration-child-manager", ["management"])
    db_session.add(manager)
    await db_session.flush()
    actors = await _child_actors(db_session)
    actor_capabilities = [actor.capabilities for actor in actors]
    for actor in actors:
        actor.capabilities = []
    await db_session.flush()
    roadmap, parent, run, version = await accepted_goal_roadmap(db_session, test_project, [
        {**roadmap_task("one", mutates_shared_state=False), "agent_id": str(builders[0].id)},
        {**roadmap_task("two", mutates_shared_state=False), "agent_id": str(builders[1].id)},
        child_item("managed-child"),
    ], budget_caps={"max_tokens": 2000, "max_turns": 40, "max_hours": 10},
       team_agent_ids=[*(agent.id for agent in builders), verifier.id, manager.id, *(agent.id for agent in actors)])
    parent.authority_model, parent.manager_user_id, parent.manager_agent_id = "agent_manager", None, manager.id
    await db_session.flush()
    service = roadmap.orchestration
    decision = {"action_type": "noop", "reason": "No integration verifier yet."}
    _local_decisions(monkeypatch, service, lambda _ctx: decision)
    for key, expected in (("one", builders[0]), ("two", builders[1])):
        await service.tick(db_session, run.id)
        row = await db_session.scalar(select(OrchestrationRoadmapItem).where(
            OrchestrationRoadmapItem.goal_id == parent.id, OrchestrationRoadmapItem.item_key == key,
        ))
        await _complete_roadmap_task(db_session, test_project.id, service, run, row, verifier.id)
        assert expected.id != verifier.id
    child_row = await db_session.scalar(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == parent.id, OrchestrationRoadmapItem.item_key == "managed-child",
    ))
    for actor, capabilities in zip(actors, actor_capabilities):
        actor.capabilities = capabilities
    await db_session.flush()
    await _complete_child(db_session, test_project, roadmap, child_row, *actors)
    await service.tick(db_session, run.id)
    await service.tick(db_session, run.id)
    gate = await _integration_gate(db_session, roadmap, parent, run, version)
    assert gate is not None
    assert set(gate.required_evidence["work_producer_agent_ids"]) == {str(agent.id) for agent in (*builders, manager)}

    for excluded in (*builders, manager):
        excluded.capabilities = [*excluded.capabilities, "validation"]
        for candidate in (*builders, manager, verifier, *actors):
            candidate.is_active = candidate.id == excluded.id
        await db_session.flush()
        with pytest.raises(HTTPException, match="No strong verification agent fit"):
            await service.execute_request_verification_action(
                db_session, run.id,
                {"action_type": "request_verification", "gate_id": str(gate.id), "work_function": "validation"},
                f"run:{run.id}:integration-excluded:{excluded.id}",
            )
        assert gate.status == "open"
    for candidate in (*builders, manager, *actors):
        candidate.is_active = False
    verifier.is_active = True
    await db_session.flush()
    decision = {"action_type": "request_verification", "gate_id": str(gate.id), "work_function": "validation"}
    await service.tick(db_session, run.id)
    await _finish_integration_verification(db_session, test_project.id, service, run, gate, verifier.id)
    decision = {"action_type": "noop", "reason": "Verification landed."}
    await service.tick(db_session, run.id)
    assert gate.status == "accepted"


async def test_accepted_integration_evidence_covers_every_parent_criterion(
    db_session, test_project, monkeypatch,
):
    builder = _agent("integration-manifest-builder", ["implementation"])
    verifier = _agent("integration-manifest-verifier", ["validation"])
    db_session.add_all([builder, verifier])
    await db_session.flush()
    roadmap, parent, run, version = await accepted_goal_roadmap(db_session, test_project, [
        {**roadmap_task("only", mutates_shared_state=False), "agent_id": str(builder.id)},
    ])
    service = roadmap.orchestration
    decision = {"action_type": "noop", "reason": "Awaiting integration verification."}
    _local_decisions(monkeypatch, service, lambda _ctx: decision)
    await service.tick(db_session, run.id)
    row = await db_session.scalar(select(OrchestrationRoadmapItem).where(OrchestrationRoadmapItem.goal_id == parent.id))
    await _complete_roadmap_task(db_session, test_project.id, service, run, row, verifier.id)
    gate = await _integration_gate(db_session, roadmap, parent, run, version)
    decision = {"action_type": "request_verification", "gate_id": str(gate.id), "work_function": "validation"}
    await service.tick(db_session, run.id)
    await _finish_integration_verification(db_session, test_project.id, service, run, gate, verifier.id)
    decision = {"action_type": "noop", "reason": "Integration accepted."}
    await service.tick(db_session, run.id)
    item_gate = await db_session.get(OrchestrationGate, row.gate_id)
    item_evidence = await db_session.scalar(select(OrchestrationEvidence).where(
        OrchestrationEvidence.gate_id == item_gate.id,
        OrchestrationEvidence.source_type == "verification",
        OrchestrationEvidence.verdict == "accepted",
    ))
    integration_evidence = await db_session.scalar(select(OrchestrationEvidence).where(
        OrchestrationEvidence.gate_id == gate.id,
        OrchestrationEvidence.source_type == "verification",
        OrchestrationEvidence.verdict == "accepted",
    ))
    assert gate.status == "accepted" and item_evidence is not None and integration_evidence is not None, {
        "failure": gate.failure_reason,
        "gate": gate.required_evidence,
        "actions": [item.request for item in await db_session.scalars(select(OrchestrationAction).where(
            OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == "request_verification",
        ))],
        "evidence": [(item.source_type, item.verdict, item.evidence_metadata) for item in await db_session.scalars(
            select(OrchestrationEvidence).where(OrchestrationEvidence.gate_id == gate.id)
        )],
    }
    manifest = roadmap.orchestration._criterion_evidence_manifest(
        [item_gate, gate], [item_evidence, integration_evidence], parent, str(version.id),
    )
    assert manifest == {
        key: [str(integration_evidence.id)]
        for key in roadmap.orchestration._declared_success_criterion_keys(parent)
    }


async def test_child_and_task_evidence_cannot_satisfy_parent_criterion(db_session, test_project, monkeypatch):
    builder = _agent("integration-no-rollup-builder", ["implementation"])
    verifier = _agent("integration-no-rollup-verifier", ["validation"])
    db_session.add_all([builder, verifier])
    await db_session.flush()
    roadmap, parent, run, _version = await accepted_goal_roadmap(db_session, test_project, [
        {**roadmap_task("only", mutates_shared_state=False), "agent_id": str(builder.id)},
    ])
    _local_decisions(monkeypatch, roadmap.orchestration, lambda _ctx: {
        "action_type": "noop", "reason": "Item verification only.",
    })
    await roadmap.orchestration.tick(db_session, run.id)
    row = await db_session.scalar(select(OrchestrationRoadmapItem).where(OrchestrationRoadmapItem.goal_id == parent.id))
    await _complete_roadmap_task(db_session, test_project.id, roadmap.orchestration, run, row, verifier.id)
    gate = await db_session.get(OrchestrationGate, row.gate_id)
    evidence = list((await db_session.scalars(select(OrchestrationEvidence).where(
        OrchestrationEvidence.gate_id == gate.id,
    ))).all())
    assert evidence and roadmap.orchestration._criterion_evidence_manifest([gate], evidence, parent) == {}


async def test_roadmap_cannot_close_without_accepted_integration_gate(db_session, test_project, monkeypatch):
    builder = _agent("integration-closeout-builder", ["implementation"])
    verifier = _agent("integration-closeout-verifier", ["validation"])
    db_session.add_all([builder, verifier])
    await db_session.flush()
    roadmap, parent, run, version = await accepted_goal_roadmap(db_session, test_project, [
        {**roadmap_task("only", mutates_shared_state=False), "agent_id": str(builder.id)},
    ])
    summarizer = _agent("integration-closeout-summarizer", ["summarization"])
    db_session.add(summarizer)
    await db_session.flush()
    _local_decisions(monkeypatch, roadmap.orchestration, lambda _ctx: {
        "action_type": "noop", "reason": "Keep integration open.",
    })
    await roadmap.orchestration.tick(db_session, run.id)
    row = await db_session.scalar(select(OrchestrationRoadmapItem).where(OrchestrationRoadmapItem.goal_id == parent.id))
    await _complete_roadmap_task(db_session, test_project.id, roadmap.orchestration, run, row, verifier.id)
    gate = await _integration_gate(db_session, roadmap, parent, run, version)
    assert gate is not None and gate.status == "open"
    for _ in range(2):
        await roadmap.orchestration.tick(db_session, run.id)
    actions = await _run_actions(db_session, run.id)
    assert not [action for action in actions if action.action_type in {"request_final_summary", "complete_run"}]


async def test_accepted_integration_reuses_existing_summary_and_closeout(
    db_session, test_project, monkeypatch, safe_effectiveness_review_continue,
):
    monkeypatch.setattr(settings, "orchestration_max_actions_per_tick", 1)
    builder = _agent("integration-golden-builder", ["implementation"])
    verifier = _agent("integration-golden-verifier", ["validation"])
    summarizer = _agent("integration-golden-summarizer", ["summarization"])
    db_session.add_all([builder, verifier, summarizer])
    await db_session.flush()
    roadmap, parent, run, version = await accepted_goal_roadmap(db_session, test_project, [
        {**roadmap_task("only", mutates_shared_state=False), "agent_id": str(builder.id)},
    ])
    parent.weight = "trivial"
    service = roadmap.orchestration
    decision = {"action_type": "noop", "reason": "Awaiting integration verification.", "wake_when": NOOP_WAKE_WHEN}
    _local_decisions(monkeypatch, service, lambda _ctx: decision)
    await service.tick(db_session, run.id)
    row = await db_session.scalar(select(OrchestrationRoadmapItem).where(OrchestrationRoadmapItem.goal_id == parent.id))
    await _complete_roadmap_task(db_session, test_project.id, service, run, row, verifier.id)
    gate = await _integration_gate(db_session, roadmap, parent, run, version)
    decision = {"action_type": "request_verification", "gate_id": str(gate.id), "work_function": "validation"}
    first = await service.tick(db_session, run.id)
    assert first["authorized_execution"]["step"] == "integration_decision"
    assert await db_session.scalar(select(OrchestrationGate.id).where(
        OrchestrationGate.run_id == run.id,
        OrchestrationGate.gate_type == "work_completed",
        OrchestrationGate.status == "open",
    )) is None
    counts = [
        await db_session.scalar(select(func.count()).select_from(model).where(model.run_id == run.id))
        for model in (OrchestrationAction, OrchestrationGate, OrchestrationEvidence)
    ]
    waiting = await service.tick(db_session, run.id)
    assert waiting["authorized_execution"] == {"step": "waiting", "reason": "waiting_active_verification"}
    assert [
        await db_session.scalar(select(func.count()).select_from(model).where(model.run_id == run.id))
        for model in (OrchestrationAction, OrchestrationGate, OrchestrationEvidence)
    ] == counts
    action, verifier_task = await _finish_integration_verification(
        db_session, test_project.id, service, run, gate, verifier.id,
    )
    decision = {"action_type": "noop", "reason": "Integration report consumed.", "wake_when": NOOP_WAKE_WHEN}
    await service.tick(db_session, run.id)
    evidence = await db_session.scalar(select(OrchestrationEvidence).where(
        OrchestrationEvidence.gate_id == gate.id,
        OrchestrationEvidence.source_type == "verification",
        OrchestrationEvidence.verdict == "accepted",
    ))
    summary_action = next(action for action in await _run_actions(db_session, run.id)
                          if action.action_type == "request_final_summary")
    summary_task = await db_session.get(Task, summary_action.target_id)
    await _record_measured_completion(
        db_session, test_project.id, run, summary_task, summarizer.id,
        json.dumps({
            "summary": "Independent integration verification accepted the Roadmap.",
            "criteria": [{"criterion_key": key, "evidence_ids": [str(evidence.id)]}
                         for key in service._declared_success_criterion_keys(parent)],
            "unresolved_gaps": [],
        }),
    )
    await service.tick(db_session, run.id)
    await service.tick(db_session, run.id)
    await db_session.refresh(parent)
    await db_session.refresh(run)
    actions = await _run_actions(db_session, run.id)
    assert (parent.status, run.status, run.phase) == ("completed", "completed", "completed")
    assert (action.target_id, verifier_task.id) == (verifier_task.id, verifier_task.id)
    assert len([item for item in actions if item.action_type == "request_verification" and item.request.get("gate_id") == str(gate.id)]) == 1
    assert len([item for item in actions if item.action_type == "request_final_summary"]) == 1
    assert len([item for item in actions if item.action_type == "complete_run"]) == 1
