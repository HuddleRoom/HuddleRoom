import asyncio
import uuid
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import patch

import pytest
import pytest_asyncio
from fastapi import HTTPException
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from huddleroom.models.artifact import Artifact
from huddleroom.models.agent import Agent
from huddleroom.models.orchestration import (
    OrchestrationAction, OrchestrationBudgetReservation, OrchestrationEvidence, OrchestrationGate,
    OrchestrationGoal, OrchestrationRoadmapItem, OrchestrationRoadmapVersion, OrchestrationRun,
    compute_goal_contract_fingerprint,
)
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
from huddleroom.models.orchestration_process import OrchestrationProcessRun
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.schemas.orchestration import GoalShapeDecision
from huddleroom.services.orchestration_goal_shape import GoalShapeResolutionError, OrchestrationGoalShapeService
from huddleroom.services.orchestration_budget_service import BudgetMeasurementError, OrchestrationBudgetService
from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
from huddleroom.services.orchestration_roadmap_service import (
    OrchestrationRoadmapService, expansion_limit_error, parse_roadmap_items,
)
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.session_service import SessionClaimAttention, SessionService
from huddleroom.schemas.session import SessionCreate
from tests.test_orchestration_runtime_e2e import _agent, _authorized_run, _seed_accepted_plan


pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def committed_roadmap_cleanup(test_engine, concurrent_sessions):
    yield
    for db in concurrent_sessions:
        await db.rollback()
    # Committed children restrict deletion of the versions in the shared cleanup.
    async with test_engine.begin() as connection:
        await connection.execute(delete(OrchestrationBudgetReservation))
        await connection.execute(delete(OrchestrationRoadmapItem))
        await connection.execute(delete(OrchestrationGoal).where(OrchestrationGoal.parent_goal_id.is_not(None)))


def goal_item(key="child", allocation=None, constraints=None):
    return {"item_key": key, "unit_type": "goal", "title": key, "objective": key,
            "success_criteria": [{"key": "local"}], "constraints": constraints or {},
            "allocation": allocation or {"max_tokens": 400, "max_turns": 1, "max_hours": 1}, "mutates_shared_state": False}


async def accepted_goal_roadmap(db, project, item, cap=1000, authority_model="human_manager", include_planner_in_team=False,
                                budget_caps=None):
    planner = _agent("child-planner", ["planning"])
    db.add(planner); await db.flush()
    service, parent, run = await _authorized_run(db, project)
    parent.authority_model = authority_model
    if authority_model == "agent_manager":
        parent.manager_user_id = None
        parent.manager_agent_id = planner.id
    parent.goal_type = "roadmap"
    parent.planning_mode = "roadmap"
    parent.budget = {"caps": budget_caps or {"max_tokens": cap, "max_turns": 10, "max_hours": 10}}
    parent.constraints = {"region": "EU"}
    parent.success_criteria = [{"key": "parent"}]
    from sqlalchemy import update
    from huddleroom.models.orchestration import OrchestrationRun, compute_goal_contract_fingerprint
    await db.execute(update(OrchestrationRun).where(OrchestrationRun.id == run.id).values(goal_contract_fingerprint=compute_goal_contract_fingerprint(parent))); await db.refresh(run)
    project.config = {"workspace_policy": {"mode": "isolated"}}
    hierarchy = await db.scalar(select(OrchestrationProcessRun).where(
        OrchestrationProcessRun.goal_id == parent.id,
        OrchestrationProcessRun.process_type == "team_hierarchy",
    ))
    hierarchy.outputs = {
        "team": "parent-team",
        **({"agent_ids": [str(planner.id)]} if include_planner_in_team else {}),
    }
    items = item if isinstance(item, list) else [item]
    artifact = Artifact(project_id=project.id, name="child-plan", artifact_type="plan", status="draft",
                        created_by_agent=planner.id, metadata_={"plan_items": items})
    db.add(artifact); await db.flush()
    roadmap = OrchestrationRoadmapService(service)
    normalized = [x.model_dump(mode="json") for x in parse_roadmap_items(
        artifact.metadata_["plan_items"], set(parent.budget["caps"])
    )]
    fingerprint = service._accepted_plan_fingerprint(normalized)
    decision = OrchestrationAuthorityDecision(goal_id=parent.id, run_id=run.id, decision_key=f"roadmap_plan:{fingerprint}",
        title="approve", question="approve", authority="human" if authority_model == "human_manager" else "manager",
        authority_agent_id=parent.manager_agent_id, status="answered", selected_option="approve",
        decided_by_user_id=parent.manager_user_id, decided_by_agent_id=parent.manager_agent_id)
    db.add(decision); await db.flush()
    version = await roadmap.accept_version(db, parent, run, artifact, {"kind": "authority_decision", "id": str(decision.id)})
    run.plan_state = {
        "status": "accepted", "accepted_artifact_id": str(artifact.id),
        "roadmap_version_id": str(version.id), "roadmap_version": version.version,
        "accepted_plan_fingerprint": version.fingerprint,
    }
    return roadmap, parent, run, version


async def release(db, project, item= None, cap=1000):
    roadmap, parent, run, version = await accepted_goal_roadmap(db, project, item or goal_item(), cap)
    result = await roadmap.advance(db, parent, run)
    row = await db.scalar(select(OrchestrationRoadmapItem).where(OrchestrationRoadmapItem.goal_id == parent.id))
    return roadmap, parent, run, result, row


async def attach_claim_lineage(db, task, run, *, agent_id=None):
    db.add(OrchestrationAction(
        run_id=run.id, idempotency_key=f"test:claim-lineage:{run.id}:{task.id}",
        action_type="create_delegation_task", request=({"agent_id": str(agent_id)} if agent_id else {}),
        target_type="task", target_id=task.id,
        status="completed",
    ))
    await db.flush()


async def test_mutable_child_without_agent_or_work_function_requires_exact_authority_then_replays(
    db_session, test_project,
):
    item = goal_item("missing-child-agent")
    item.update({
        "mutates_shared_state": True,
        "staging_boundary": {"type": "git_worktree", "identifier": "missing", "reversible": True},
    })
    roadmap, parent, run, version = await accepted_goal_roadmap(db_session, test_project, item)

    assert await roadmap.advance(db_session, parent, run) == {
        "step": "waiting", "reason": "waiting_unstaged_approval",
    }
    assert await db_session.scalar(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == parent.id,
    )) is None
    decision = await db_session.scalar(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.goal_id == parent.id,
        OrchestrationAuthorityDecision.run_id == run.id,
        OrchestrationAuthorityDecision.decision_key == f"roadmap_unstaged_mutation:{version.id}:missing-child-agent",
    ))
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, decision, selected_option="approve", decided_by_user_id=parent.manager_user_id,
    )
    released = await roadmap.advance(db_session, parent, run)
    assert released == {"step": "release_item", "item_key": "missing-child-agent", "unit_type": "goal"}
    counts = [
        await db_session.scalar(select(func.count()).select_from(model))
        for model in (OrchestrationRoadmapItem, OrchestrationBudgetReservation, OrchestrationAction)
    ]
    assert await roadmap.advance(db_session, parent, run) == {
        "step": "waiting", "reason": "waiting_active_work",
    }
    assert [
        await db_session.scalar(select(func.count()).select_from(model))
        for model in (OrchestrationRoadmapItem, OrchestrationBudgetReservation, OrchestrationAction)
    ] == counts


async def test_child_planning_task_reassignment_and_metadata_tamper_cannot_change_claim_agent(
    db_session, test_project,
):
    item = goal_item("tamper-child")
    item.update({"mutates_shared_state": True, "staging_boundary": None})
    roadmap, parent, run, version = await accepted_goal_roadmap(
        db_session, test_project, item, include_planner_in_team=True,
    )
    await roadmap.advance(db_session, parent, run)
    approval = await db_session.scalar(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.goal_id == parent.id,
        OrchestrationAuthorityDecision.decision_key == f"roadmap_unstaged_mutation:{version.id}:tamper-child",
    ))
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, approval, selected_option="approve", decided_by_user_id=parent.manager_user_id,
    )
    assert (await roadmap.advance(db_session, parent, run))["step"] == "release_item"
    row = await db_session.scalar(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == parent.id,
    ))
    child = await db_session.get(OrchestrationGoal, row.child_goal_id)
    child_run = await db_session.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == child.id))
    outsider = _agent("child-immutable-outsider", ["planning"])
    db_session.add(outsider)
    await db_session.flush()
    planner_id = uuid.UUID(child.orchestrator_context["team"]["agent_ids"][0])
    planner = await db_session.get(Agent, planner_id)
    planner.capabilities = ["planning"]
    action = await roadmap.orchestration.execute_request_plan_action(
        db_session, child_run.id,
        {"action_type": "request_plan", "agent_id": str(planner.id), "work_function": "planning", "scope": "Plan child."},
        f"run:{child_run.id}:kind:tamper-child-plan",
    )
    task = await db_session.get(Task, action.target_id)
    task.assigned_to = outsider.id
    task.metadata_ = {}

    with pytest.raises(HTTPException, match="immutable assigned team agent"):
        await SessionService().create(db_session, SessionCreate(
            agent_id=outsider.id, task_id=task.id, project_id=test_project.id,
        ))


async def conserved(db, parent):
    summary = await OrchestrationBudgetService().remaining(db, parent)
    for dimension, cap in summary["caps"].items():
        assert Decimal(summary["direct_spend"][dimension]) + Decimal(summary["settled_child_spend"][dimension]) + Decimal(summary["active_reservations"][dimension]) + Decimal(summary["active_commitments"][dimension]) + Decimal(summary["remaining"][dimension]) == Decimal(cap)
    return summary


async def assert_child_release(db, parent, run, version, item, *, failed_item_key=None):
    """Assert exact persisted cardinality, deterministic identities, and bindings."""
    key = item["item_key"]
    child_id = uuid.uuid5(uuid.NAMESPACE_URL, f"rally:roadmap-child:{parent.id}:{key}")
    child_run_id = uuid.uuid5(uuid.NAMESPACE_URL, f"rally:roadmap-child-run:{parent.id}:{key}")
    lineage_id = uuid.uuid5(uuid.NAMESPACE_URL, f"rally:roadmap-item:{parent.id}:{key}")
    gate_id = uuid.uuid5(uuid.NAMESPACE_URL, f"rally:orchestration:gate:{run.id}:roadmap_item:{version.id}:{key}")
    reservation_id = uuid.uuid5(uuid.NAMESPACE_URL, f"rally:budget-reservation:{parent.id}:{key}")
    auth_id = uuid.uuid5(uuid.NAMESPACE_URL, f"rally:authorize-child:{child_run_id}")
    queries = [
        (OrchestrationGoal, OrchestrationGoal.parent_goal_id == parent.id, child_id),
        (OrchestrationRun, OrchestrationRun.goal_id == child_id, child_run_id),
        (OrchestrationRoadmapItem, OrchestrationRoadmapItem.goal_id == parent.id, lineage_id),
        (OrchestrationGate, (OrchestrationGate.run_id == run.id) &
         (OrchestrationGate.gate_type == "child_goal_completed"), gate_id),
        (OrchestrationBudgetReservation, OrchestrationBudgetReservation.parent_goal_id == parent.id, reservation_id),
        (OrchestrationAction, OrchestrationAction.run_id == child_run_id, auth_id),
    ]
    rows = []
    for model, condition, expected_id in queries:
        matches = list(await db.scalars(select(model).where(condition)))
        assert [row.id for row in matches] == [expected_id]
        rows.append(matches[0])
    child, child_run, lineage, gate, reservation, auth = rows
    assert child.goal_type == "outcome" and child.status == "active"
    assert child.project_id == parent.project_id
    assert (child.roadmap_version_id, child.roadmap_item_key) == (version.id, key)
    assert child.budget == {"caps": {key: str(value) for key, value in item["allocation"].items()}}
    assert child.success_criteria == item["success_criteria"]
    assert (child.authority_model, child.manager_agent_id, child.manager_user_id) == (
        parent.authority_model, parent.manager_agent_id, parent.manager_user_id,
    )
    assert (child_run.phase, child_run.status) == ("authorized", "running")
    assert (child.goal_type_source, child.planning_mode) == ("parent_explicit", "implementation")
    assert key in child.goal_type_rationale and child.planning_rationale.strip()
    assert (child_run.goal_type_snapshot, child_run.planning_mode_snapshot) == ("outcome", "implementation")
    assert child_run.goal_contract_fingerprint == compute_goal_contract_fingerprint(child)
    assert child_run.supersedes_run_id is None
    assert child_run.plan_state == {"child_delta_baseline": {
        "status": "accepted", "snapshot_version": 1, "parent_goal_id": str(parent.id),
        "roadmap_version_id": str(version.id), "roadmap_item_key": key,
        "approval_reference": version.approval_reference,
    }}
    assert (lineage.first_version_id, lineage.item_key, lineage.unit_type) == (version.id, key, "goal")
    assert (lineage.child_goal_id, lineage.gate_id, lineage.task_id) == (child_id, gate_id, None)
    assert lineage.item_snapshot == version.snapshot["items"][0]
    assert lineage.completed_at is None
    assert (gate.status, gate.success_criterion_key) == ("open", f"roadmap_item:{key}")
    assert gate.required_evidence == {
        "roadmap_item_key": key, "roadmap_version_id": str(version.id),
        "required_source_types": ["child_goal"], "min_count": 1,
    }
    assert (reservation.roadmap_item_id, reservation.child_goal_id) == (lineage_id, child_id)
    assert reservation.status == "active" and reservation.allocation == child.budget["caps"]
    assert reservation.settled_spend == {} and reservation.settled_at is None
    assert (auth.action_type, auth.status, auth.target_type, auth.target_id) == (
        "authorize_execution", "completed", "run", child_run_id,
    )
    assert auth.idempotency_key == f"run:{child_run_id}:kind:authorize_execution"
    assert auth.request == {"action_type": "authorize_execution", "actor": f"roadmap_parent:{parent.id}"}
    releases = list(await db.scalars(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == "release_roadmap_item",
    )))
    assert len(releases) == (2 if failed_item_key else 1)
    if failed_item_key:
        failed = [action for action in releases if action.request["item_key"] == failed_item_key]
        assert len(failed) == 1
        assert failed[0].status == "failed" and failed[0].target_id is None
        assert failed[0].idempotency_key == f"run:{run.id}:kind:release_roadmap_item:{failed_item_key}"
    action, = [action for action in releases if action.request["item_key"] == key]
    assert action.idempotency_key == f"run:{run.id}:kind:release_roadmap_item:{key}"
    assert (action.status, action.target_type, action.target_id) == ("completed", "goal", child_id)
    assert action.request == {"item_key": key, "unit_type": "goal"}
    return tuple(row.id for row in rows) + (action.id,)


async def test_goal_item_reservation_child_run_and_gate_are_atomic(db_session, test_project):
    item = goal_item()
    roadmap, parent, run, version = await accepted_goal_roadmap(db_session, test_project, item)
    result = await roadmap.orchestration.tick(db_session, run.id)
    assert result["authorized_execution"] == {"step": "release_item", "item_key": "child", "unit_type": "goal"}
    await assert_child_release(db_session, parent, run, version, item)
    await conserved(db_session, parent)


async def test_goal_item_replay_returns_one_child_and_reservation(db_session, test_project):
    item = goal_item()
    roadmap, parent, run, version = await accepted_goal_roadmap(db_session, test_project, item)
    await roadmap.orchestration.tick(db_session, run.id)
    identities = await assert_child_release(db_session, parent, run, version, item)
    await roadmap.orchestration.tick(db_session, run.id)
    assert await assert_child_release(db_session, parent, run, version, item) == identities
    parsed = parse_roadmap_items([item], set(parent.budget["caps"]))[0]
    assert await roadmap.release_goal_item(db_session, parent, run, version, parsed) == {
        "step": "release_item", "item_key": "child", "unit_type": "goal",
    }
    assert await assert_child_release(db_session, parent, run, version, item) == identities


async def test_goal_item_failure_after_reservation_rolls_back_everything(db_session, test_project, monkeypatch):
    roadmap, parent, run, version = await accepted_goal_roadmap(db_session, test_project, goal_item())

    async def fail_event(*args, **kwargs):
        raise RuntimeError("injected event failure")

    monkeypatch.setattr("huddleroom.services.orchestration_roadmap_service.emit_event_once", fail_event)
    with pytest.raises(RuntimeError, match="injected event failure"):
        await roadmap.advance(db_session, parent, run)
    assert await db_session.scalar(select(OrchestrationRoadmapItem).where(OrchestrationRoadmapItem.goal_id == parent.id)) is None
    assert await db_session.scalar(select(OrchestrationGoal).where(OrchestrationGoal.parent_goal_id == parent.id)) is None
    assert await db_session.scalar(select(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.parent_goal_id == parent.id)) is None
    child_run_id = uuid.uuid5(uuid.NAMESPACE_URL, f"rally:roadmap-child-run:{parent.id}:child")
    assert await db_session.get(OrchestrationRun, child_run_id) is None
    assert await db_session.get(OrchestrationGate, uuid.uuid5(
        uuid.NAMESPACE_URL, f"rally:orchestration:gate:{run.id}:roadmap_item:{version.id}:child")) is None
    assert await db_session.get(OrchestrationAction, uuid.uuid5(
        uuid.NAMESPACE_URL, f"rally:authorize-child:{child_run_id}")) is None


@pytest.mark.parametrize("authority_model", ["human_manager", "agent_manager"])
async def test_child_contract_preserves_parent_authority_team_criteria_and_workspace(
    db_session, test_project, authority_model,
):
    item = goal_item(constraints={"language": "Python"})
    roadmap, parent, run, version = await accepted_goal_roadmap(
        db_session, test_project, item, authority_model=authority_model,
    )
    parent.budget = {**parent.budget, "policy": {"allow_expansion": False}}
    expected = {
        "snapshot_version": 1, "parent_goal_id": str(parent.id),
        "roadmap_version_id": str(version.id), "roadmap_item_key": "child",
        "objective": "Ship the authorized-execution runtime loop",
        "constraints": {"region": "EU"},
        "budget_policy": {"caps": {"max_tokens": 1000, "max_turns": 10, "max_hours": 10},
                          "policy": {"allow_expansion": False}},
        "authority": {"authority_model": authority_model,
                      "manager_agent_id": str(parent.manager_agent_id) if parent.manager_agent_id else None,
                      "manager_user_id": str(parent.manager_user_id) if parent.manager_user_id else None},
            "team": {"team": "parent-team", "agent_ids": []},
        "success_criteria": [{"key": "parent"}],
        "workspace_policy": {"mode": "isolated"},
    }
    assert expected["authority"]["manager_user_id" if authority_model == "human_manager" else "manager_agent_id"]
    await roadmap.advance(db_session, parent, run)
    child_id, *_ = await assert_child_release(db_session, parent, run, version, item)
    child = await db_session.get(OrchestrationGoal, child_id)
    assert child.parent_contract_snapshot == expected
    assert child.objective == "child" and child.constraints == {"region": "EU", "language": "Python"}
    assert child.success_criteria == [{"key": "local"}]
    delta = {"objective": "child", "constraints": {"language": "Python"}, "success_criteria": [{"key": "local"}]}
    assert child.goal_delta == delta
    assert child.budget == {"caps": {"max_tokens": "400", "max_turns": "1", "max_hours": "1"}}
    parent.objective = "Changed parent objective"
    parent.constraints = {"region": "US"}
    parent.budget = {"caps": {"max_tokens": 900}}
    parent.success_criteria = [{"key": "changed"}]
    parent.authority_model, parent.manager_agent_id, parent.manager_user_id = "no_manager", None, None
    test_project.config = {"workspace_policy": {"mode": "shared"}}
    hierarchy = await db_session.scalar(select(OrchestrationProcessRun).where(
        OrchestrationProcessRun.goal_id == parent.id, OrchestrationProcessRun.process_type == "team_hierarchy",
    ))
    hierarchy.outputs = {"team": "changed"}
    await db_session.flush()
    await db_session.refresh(child)
    assert child.parent_contract_snapshot == expected and child.goal_delta == delta
    with pytest.raises(ValueError, match="immutable"):
        child.parent_contract_snapshot = {**deepcopy(expected), "objective": "overwrite"}


async def test_child_delta_rejects_conflicting_parent_constraint(db_session, test_project, concurrent_sessions):
    roadmap, parent, run, version = await accepted_goal_roadmap(db_session, test_project, goal_item(constraints={"region": "US"}))
    parent_id, run_id = parent.id, run.id
    await db_session.commit()
    writer, reader = concurrent_sessions
    for _ in range(2):
        assert (await roadmap.orchestration.tick(writer, run_id))["authorized_execution"] == {"step": "needs_attention"}
        assert not writer.in_transaction()
    persisted_run = await reader.get(OrchestrationRun, run_id)
    assert [x for x in persisted_run.active_blockers if x["kind"] == "child_contract"] == [{
        "kind": "child_contract", "item_key": "child", "roadmap_version_id": str(version.id),
        "reason": "Child constraint conflicts with parent: region",
    }]
    for model, condition in [
        (OrchestrationRoadmapItem, OrchestrationRoadmapItem.goal_id == parent_id),
        (OrchestrationGoal, OrchestrationGoal.parent_goal_id == parent_id),
        (OrchestrationBudgetReservation, OrchestrationBudgetReservation.parent_goal_id == parent_id),
    ]:
        assert list(await reader.scalars(select(model).where(condition))) == []


async def test_concurrent_goal_item_release_cannot_overallocate(
    db_session, test_project, concurrent_sessions, test_engine, monkeypatch, committed_roadmap_cleanup,
):
    roadmap, parent, run, version = await accepted_goal_roadmap(
        db_session, test_project, [goal_item("one", {"max_tokens": 700, "max_turns": 1, "max_hours": 1}),
                                  goal_item("two", {"max_tokens": 700, "max_turns": 1, "max_hours": 1})], 1400)
    parent.budget = {"caps": {"max_tokens": 1000, "max_turns": 10, "max_hours": 10}}
    parent_id, run_id, version_id = parent.id, run.id, version.id
    await db_session.commit()
    first, second = concurrent_sessions
    assert first is not second
    both_attempting = asyncio.Event()
    attempts, holders, lock_events = [], set(), []
    real_lock = OrchestrationService._lock_goal_for_baseline_transition

    @asynccontextmanager
    async def observed_lock(service, db, goal_id, *, tick_owns_transaction=False):
        assert goal_id == parent_id
        attempts.append(db)
        if len(attempts) == 2:
            both_attempting.set()
        async with real_lock(service, db, goal_id, tick_owns_transaction=tick_owns_transaction):
            assert not holders, "Both ticks entered the parent critical section"
            holders.add(db)
            lock_events.append(("enter", db))
            try:
                # Hold the actual production lock until the other tick attempts it.
                await asyncio.wait_for(both_attempting.wait(), timeout=10)
                yield
                assert not db.in_transaction(), "Winning tick must commit while holding the parent lock"
            finally:
                lock_events.append(("exit", db))
                holders.remove(db)

    monkeypatch.setattr(OrchestrationService, "_lock_goal_for_baseline_transition", observed_lock)

    async def tick(db):
        try:
            result = await OrchestrationService().tick(db, run_id)
            assert not db.in_transaction(), "Winning tick must commit before releasing the parent lock"
            return result
        except HTTPException as exc:
            await db.rollback()  # Same exception boundary as the production database dependency.
            return exc

    results = await asyncio.wait_for(asyncio.gather(tick(first), tick(second)), timeout=30)
    assert set(attempts) == {first, second} and len(attempts) == 2
    winner, loser = attempts
    assert lock_events == [("enter", winner), ("exit", winner), ("enter", loser), ("exit", loser)]
    successes = [result for result in results if isinstance(result, dict)]
    assert len(successes) == 2
    assert {tuple(result["authorized_execution"].items()) for result in successes} == {
        (("step", "release_item"), ("item_key", "one"), ("unit_type", "goal")),
        (("step", "waiting"), ("reason", "budget_wait")),
    }
    monkeypatch.setattr(OrchestrationService, "_lock_goal_for_baseline_transition", real_lock)
    replay = await tick(loser)
    assert replay["authorized_execution"] == {"step": "waiting", "reason": "budget_wait"}
    async with async_sessionmaker(test_engine, expire_on_commit=False)() as reader:
        parent = await reader.get(OrchestrationGoal, parent_id)
        run = await reader.get(OrchestrationRun, run_id)
        version = await roadmap.current_version(reader, parent_id)
        assert version.id == version_id
        await assert_child_release(reader, parent, run, version, goal_item(
            "one", {"max_tokens": 700, "max_turns": 1, "max_hours": 1},
        ))
        summary = await conserved(reader, parent)
        assert summary == {
            "caps": {"max_tokens": "1000", "max_turns": "10", "max_hours": "10"},
            "direct_spend": {"max_tokens": "0", "max_turns": "0", "max_hours": "0"},
            "settled_child_spend": {"max_tokens": "0", "max_turns": "0", "max_hours": "0"},
            "active_reservations": {"max_tokens": "700", "max_turns": "1", "max_hours": "1"},
            "active_commitments": {"max_tokens": "0", "max_turns": "0", "max_hours": "0"},
            "remaining": {"max_tokens": "300", "max_turns": "9", "max_hours": "9"},
        }
        assert all(Decimal(amount) >= 0 for amount in summary["remaining"].values())


async def test_child_settlement_releases_unused_allocation(db_session, test_project):
    roadmap, parent, run, _, row = await release(db_session, test_project)
    child = await db_session.get(OrchestrationGoal, row.child_goal_id)
    child_run = await db_session.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == child.id))
    agent = _agent("child-spend", ["implementation"])
    db_session.add(agent); await db_session.flush()
    task = Task(project_id=test_project.id, title="spent", status="done",
        metadata_={"orchestration": {"run_id": str(child_run.id)}})
    db_session.add(task); await db_session.flush()
    await attach_claim_lineage(db_session, task, child_run)
    db_session.add(Session(task_id=task.id, agent_id=agent.id, project_id=test_project.id, adapter_type="test",
        status="completed", started_at=child_run.started_at, ended_at=child_run.started_at + timedelta(hours=1),
        metadata_={"token_count_in": 100, "token_count_out": 150, "token_usage_complete": False,
                   "_roadmap_cli_budget_approval": {"decision_id": str(uuid.uuid4())},
                   "_roadmap_cli_token_grants": [300]}))
    child.status = "completed"
    await roadmap.advance(db_session, parent, run)
    reservation = await db_session.scalar(select(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.roadmap_item_id == row.id))
    assert reservation.status == "settled" and reservation.settled_spend == {
        "max_tokens": "300", "max_turns": "1", "max_hours": "1"}
    await conserved(db_session, parent)


async def test_missing_required_usage_measurement_blocks_settlement(db_session, test_project):
    roadmap, parent, run, _, row = await release(db_session, test_project)
    child = await db_session.get(OrchestrationGoal, row.child_goal_id)
    child_run = await db_session.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == child.id))
    agent = _agent("child-measurement", ["implementation"])
    db_session.add(agent); await db_session.flush()
    task = Task(project_id=test_project.id, title="child", status="done", metadata_={"orchestration": {"run_id": str(child_run.id)}})
    db_session.add(task); await db_session.flush()
    await attach_claim_lineage(db_session, task, child_run)
    session = Session(task_id=task.id, agent_id=agent.id, project_id=test_project.id,
        adapter_type="test", status="completed", started_at=child_run.started_at,
        ended_at=child_run.started_at + timedelta(hours=1), metadata_={})
    db_session.add(session)
    child.status = "completed"
    assert await roadmap.advance(db_session, parent, run) == {"step": "waiting", "reason": "needs_attention"}
    reservation = await db_session.scalar(select(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.roadmap_item_id == row.id))
    assert reservation.status == "active" and any(x["kind"] == "budget_measurement" for x in run.active_blockers)
    session.metadata_ = {"token_count_in": 100, "token_count_out": 150}
    await roadmap.advance(db_session, parent, run)
    assert reservation.status == "settled"
    assert not any(x["kind"] == "budget_measurement" for x in run.active_blockers)


async def test_active_child_missing_claim_measurement_waits_and_recovers_through_tick(
    db_session, test_project, monkeypatch,
):
    roadmap, parent, parent_run, _, row = await release(db_session, test_project)
    child = await db_session.get(OrchestrationGoal, row.child_goal_id)
    child_run = await db_session.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == child.id))
    parent_run.status = "paused"
    agent = _agent("active-child-missing-claim", ["implementation"])
    db_session.add(agent); await db_session.flush()
    task = Task(project_id=test_project.id, title="active child", status="in_progress", assigned_to=agent.id,
        metadata_={"orchestration": {"run_id": str(child_run.id)}})
    db_session.add(task); await db_session.flush()
    await attach_claim_lineage(db_session, task, child_run)
    session = Session(task_id=task.id, agent_id=agent.id, project_id=test_project.id,
        adapter_type="api", status="running", metadata_={})
    db_session.add(session); await db_session.flush()

    async def must_not_decide(*_args, **_kwargs):
        raise AssertionError("missing claim telemetry must wait before dispatch")

    monkeypatch.setattr(roadmap.orchestration, "request_llm_decision", must_not_decide)
    waiting = await roadmap.orchestration.tick(db_session, child_run.id)
    assert waiting["authorized_execution"] == {"step": "needs_attention"}
    assert any(blocker["kind"] == "budget_measurement" for blocker in child_run.active_blockers)

    session.metadata_ = {"_run_config": {"max_tokens": 400, "timeout": 3600}}
    recovered = await roadmap.orchestration._sync_measured_budget_exhaustion(db_session, child, child_run)
    assert recovered is False
    assert not any(blocker["kind"] == "budget_measurement" for blocker in child_run.active_blockers)


async def test_cancel_with_missing_measurement_conservatively_settles_full_allocation(db_session, test_project):
    roadmap, parent, run, _, row = await release(db_session, test_project)
    child = await db_session.get(OrchestrationGoal, row.child_goal_id)
    child_run = await db_session.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == child.id))
    agent = _agent("child-cancel", ["implementation"])
    db_session.add(agent); await db_session.flush()
    task = Task(project_id=test_project.id, title="cancelled", status="cancelled",
        metadata_={"orchestration": {"run_id": str(child_run.id)}})
    db_session.add(task); await db_session.flush()
    await attach_claim_lineage(db_session, task, child_run)
    db_session.add(Session(task_id=task.id, agent_id=agent.id, project_id=test_project.id,
        adapter_type="test", status="cancelled", metadata_={}))
    child.status = "cancelled"
    await roadmap.advance(db_session, parent, run)
    reservation = await db_session.scalar(select(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.roadmap_item_id == row.id))
    assert reservation.status == "settled" and reservation.measurement_complete is False
    assert reservation.settled_spend == {"max_tokens": "400", "max_turns": "1", "max_hours": "1"}
    await conserved(db_session, parent)


async def test_cancelled_child_retains_measured_overage_and_blocks_parent(db_session, test_project):
    roadmap, parent, run, _, row = await release(db_session, test_project)
    child = await db_session.get(OrchestrationGoal, row.child_goal_id)
    child_run = await db_session.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == child.id))
    agent = _agent("child-cancel-overage", ["implementation"])
    db_session.add(agent); await db_session.flush()
    task = Task(project_id=test_project.id, title="cancelled overage", status="cancelled",
        metadata_={"orchestration": {"run_id": str(child_run.id)}})
    db_session.add(task); await db_session.flush()
    await attach_claim_lineage(db_session, task, child_run)
    db_session.add(Session(task_id=task.id, agent_id=agent.id, project_id=test_project.id,
        adapter_type="test", status="cancelled", started_at=child_run.started_at,
        ended_at=child_run.started_at + timedelta(hours=1), metadata_={"token_count_in": 450, "token_count_out": 0}))
    child.status = "cancelled"
    assert await roadmap.advance(db_session, parent, run) == {"step": "waiting", "reason": "needs_attention"}
    reservation = await db_session.scalar(select(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.roadmap_item_id == row.id))
    assert reservation.settled_spend["max_tokens"] == "450"
    assert any(blocker["kind"] == "budget_integrity" for blocker in run.active_blockers)


async def test_cancelled_child_keeps_known_overage_when_another_session_lacks_telemetry(db_session, test_project):
    roadmap, parent, run, _, row = await release(db_session, test_project)
    child = await db_session.get(OrchestrationGoal, row.child_goal_id)
    child_run = await db_session.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == child.id))
    agent = _agent("child-cancel-mixed-telemetry", ["implementation"])
    db_session.add(agent); await db_session.flush()
    measured = Task(project_id=test_project.id, title="measured", status="cancelled",
        metadata_={"orchestration": {"run_id": str(child_run.id)}})
    missing = Task(project_id=test_project.id, title="missing", status="cancelled",
        metadata_={"orchestration": {"run_id": str(child_run.id)}})
    db_session.add_all([measured, missing]); await db_session.flush()
    await attach_claim_lineage(db_session, measured, child_run)
    await attach_claim_lineage(db_session, missing, child_run)
    db_session.add_all([
        Session(task_id=measured.id, agent_id=agent.id, project_id=test_project.id,
            adapter_type="test", status="cancelled", started_at=child_run.started_at,
            ended_at=child_run.started_at + timedelta(hours=1), metadata_={
                "token_count_in": 450, "token_count_out": 0, "token_usage_complete": False,
                "_roadmap_cli_budget_approval": {"decision_id": str(uuid.uuid4())},
                "_roadmap_cli_token_grants": [500],
            }),
        Session(task_id=missing.id, agent_id=agent.id, project_id=test_project.id,
            adapter_type="test", status="cancelled", metadata_={}),
    ])
    child.status = "cancelled"

    assert await roadmap.advance(db_session, parent, run) == {"step": "waiting", "reason": "needs_attention"}
    reservation = await db_session.scalar(select(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.roadmap_item_id == row.id))
    assert reservation.measurement_complete is False
    assert reservation.settled_spend["max_tokens"] == "500"


async def test_parent_active_session_commitment_blocks_child_budget_admission(db_session, test_project):
    roadmap, parent, run, version = await accepted_goal_roadmap(
        db_session, test_project, goal_item("child", {"max_tokens": 400, "max_turns": 1, "max_hours": 1}), cap=1000,
    )
    agent = _agent("parent-active-budget", ["implementation"])
    db_session.add(agent); await db_session.flush()
    task = Task(project_id=test_project.id, title="active parent work", status="in_progress",
        metadata_={"orchestration": {"run_id": str(run.id)}})
    db_session.add(task); await db_session.flush()
    await attach_claim_lineage(db_session, task, run)
    db_session.add(Session(task_id=task.id, agent_id=agent.id, project_id=test_project.id,
        adapter_type="test", status="running", metadata_={"_run_config": {"max_tokens": 700, "timeout": 3600}}))
    summary = await OrchestrationBudgetService().remaining(db_session, parent)
    assert summary["active_commitments"] == {"max_tokens": "700", "max_turns": "1", "max_hours": "1"}
    parsed = parse_roadmap_items(version.snapshot["items"], set(parent.budget["caps"]))[0]
    with pytest.raises(HTTPException, match="remaining parent budget"):
        await roadmap.release_goal_item(db_session, parent, run, version, parsed)


async def test_child_session_claim_clamps_adapter_limit_to_remaining_cap(db_session, test_project):
    roadmap, parent, parent_run, _version = await accepted_goal_roadmap(
        db_session, test_project, goal_item("child", {"max_tokens": 400, "max_turns": 1, "max_hours": 1}),
        include_planner_in_team=True,
    )
    await roadmap.advance(db_session, parent, parent_run)
    row = await db_session.scalar(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == parent.id,
    ))
    child_run = await db_session.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == row.child_goal_id))
    child = await db_session.get(OrchestrationGoal, row.child_goal_id)
    agent = await db_session.get(Agent, uuid.UUID(child.orchestrator_context["team"]["agent_ids"][0]))
    agent.capabilities = ["implementation"]
    task = Task(project_id=test_project.id, title="child work", status="ready", assigned_to=agent.id,
        metadata_={"orchestration": {"run_id": str(child_run.id)}})
    db_session.add(task); await db_session.flush()
    await attach_claim_lineage(db_session, task, child_run, agent_id=agent.id)
    session = await SessionService().create(db_session, SessionCreate(
        agent_id=agent.id, task_id=task.id, project_id=test_project.id, max_tokens=1000,
    ))
    assert session.metadata_["_run_config"] == {
        "max_tokens": 400, "timeout": 3600, "_roadmap_budget_enforced": True,
    }


async def test_cli_token_claim_waits_for_exact_human_budget_authority_then_releases_once(
    db_session, test_project,
):
    roadmap, parent, parent_run, version = await accepted_goal_roadmap(
        db_session, test_project, goal_item("cli-budget", {"max_tokens": 400, "max_turns": 1, "max_hours": 1}),
        include_planner_in_team=True,
    )
    await roadmap.advance(db_session, parent, parent_run)
    row = await db_session.scalar(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == parent.id,
    ))
    child = await db_session.get(OrchestrationGoal, row.child_goal_id)
    child_run = await db_session.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == child.id))
    agent = await db_session.get(Agent, uuid.UUID(child.orchestrator_context["team"]["agent_ids"][0]))
    agent.adapter_type = "cli"
    task = Task(project_id=test_project.id, title="cli budget", status="ready", assigned_to=agent.id)
    db_session.add(task); await db_session.flush()
    await attach_claim_lineage(db_session, task, child_run, agent_id=agent.id)

    with pytest.raises(HTTPException, match="CLI budget authority"):
        await SessionService().create(db_session, SessionCreate(
            agent_id=agent.id, task_id=task.id, project_id=test_project.id,
        ))
    decision = await db_session.scalar(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.decision_key == (
        f"roadmap_budget_adapter:{version.id}:cli-budget:%:cli:max_tokens"
        ),
    ))
    assert decision is None
    assert await db_session.scalar(select(func.count(Session.id))) == 0


@pytest.mark.parametrize("answer", ["approve", "reject"])
async def test_child_cli_planning_authority_replays_pending_then_resolves_once(
    db_session, test_project, answer,
):
    """An admitted child's planning claim uses the parent-bound CLI authority exactly once."""
    roadmap, parent, parent_run, version = await accepted_goal_roadmap(
        db_session, test_project,
        goal_item("child-cli", {"max_tokens": 400, "max_turns": 1, "max_hours": 1}),
        include_planner_in_team=True,
    )
    await roadmap.advance(db_session, parent, parent_run)
    row = await db_session.scalar(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == parent.id,
    ))
    child = await db_session.get(OrchestrationGoal, row.child_goal_id)
    child_run = await db_session.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == child.id))
    agent = await db_session.get(Agent, uuid.UUID(child.orchestrator_context["team"]["agent_ids"][0]))
    agent.adapter_type = "cli"
    context = {"roadmap": {"roadmap_version_id": str(version.id), "roadmap_item_key": row.item_key}}

    assert (await roadmap.orchestration.roadmap_pre_release_authority(
        db_session, child_run.id, agent, context,
    ))["status"] == "pending"
    decision = await db_session.scalar(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.goal_id == parent.id,
        OrchestrationAuthorityDecision.decision_key.like("roadmap_budget_adapter:%:child-cli:%:cli:max_tokens"),
    ))
    assert (await roadmap.orchestration.roadmap_pre_release_authority(
        db_session, child_run.id, agent, context,
    ))["status"] == "pending"
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.decision_key == decision.decision_key,
    )) == 1

    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, decision, selected_option=answer, decided_by_user_id=parent.manager_user_id,
    )
    resolved = await roadmap.orchestration.roadmap_pre_release_authority(
        db_session, child_run.id, agent, context,
    )
    assert resolved["status"] == ("approved" if answer == "approve" else "rejected")
    if answer == "reject":
        blockers = [item for item in parent_run.active_blockers if item.get("decision_key") == decision.decision_key]
        assert len(blockers) == 1 and blockers[0]["kind"] == "budget_integrity"


@pytest.mark.parametrize("answer", ["approve", "reject"])
async def test_accepted_child_plan_expansion_waits_before_creating_lineage(
    db_session, test_project, answer,
):
    roadmap, parent, parent_run, _version = await accepted_goal_roadmap(
        db_session, test_project,
        goal_item("accepted-child-cli", {"max_tokens": 400, "max_turns": 2, "max_hours": 1}),
        include_planner_in_team=True,
    )
    await roadmap.advance(db_session, parent, parent_run)
    row = await db_session.scalar(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == parent.id,
    ))
    child = await db_session.get(OrchestrationGoal, row.child_goal_id)
    child_run = await db_session.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == child.id))
    agent = await db_session.get(Agent, uuid.UUID(child.orchestrator_context["team"]["agent_ids"][0]))
    agent.adapter_type = "api"
    await _seed_accepted_plan(
        db_session, test_project, roadmap.orchestration, child_run, agent,
        [{"id": "cli-work", "work_function": "implementation", "scope": "child work",
          "deliverable": "verified", "agent_id": str(agent.id)}],
    )
    agent.adapter_type = "cli"
    request = {"action_type": "expand_plan_item", "plan_item_id": "cli-work", "work_function": "implementation"}
    key = roadmap.orchestration._expand_plan_item_idempotency_key(child_run.id, "cli-work")

    for _ in range(2):
        with pytest.raises(HTTPException, match="budget_wait"):
            await roadmap.orchestration.execute_expand_plan_item_action(db_session, child_run.id, request, key)
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
        OrchestrationAction.run_id == child_run.id,
        OrchestrationAction.action_type.in_(("expand_plan_item", "create_delegation_task")),
    )) == 0
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationGate).where(
        OrchestrationGate.run_id == child_run.id,
        OrchestrationGate.success_criterion_key == "plan_item:cli-work",
    )) == 0
    decision = await db_session.scalar(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.goal_id == parent.id,
        OrchestrationAuthorityDecision.decision_key.like("roadmap_budget_adapter:%:accepted-child-cli:%:cli:max_tokens"),
    ))
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, decision, selected_option=answer, decided_by_user_id=parent.manager_user_id,
    )
    if answer == "reject":
        with pytest.raises(HTTPException, match="needs_attention"):
            await roadmap.orchestration.execute_expand_plan_item_action(db_session, child_run.id, request, key)
        assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
            OrchestrationAction.run_id == child_run.id,
            OrchestrationAction.action_type.in_(("expand_plan_item", "create_delegation_task")),
        )) == 0
        return
    action = await roadmap.orchestration.execute_expand_plan_item_action(db_session, child_run.id, request, key)
    assert action.status == "completed"
    replay = await roadmap.orchestration.execute_expand_plan_item_action(db_session, child_run.id, request, key)
    assert replay.id == action.id
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
        OrchestrationAction.run_id == child_run.id,
        OrchestrationAction.action_type.in_(("expand_plan_item", "create_delegation_task")),
    )) == 2


async def test_cli_max_hours_only_claim_needs_no_human_budget_authority(db_session, test_project):
    roadmap, parent, parent_run, _ = await accepted_goal_roadmap(
        db_session, test_project, goal_item("cli-hours", {"max_hours": 1, "max_turns": 1}), cap=1000,
        include_planner_in_team=True, budget_caps={"max_hours": 1, "max_turns": 1},
    )
    await roadmap.advance(db_session, parent, parent_run)
    row = await db_session.scalar(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == parent.id,
    ))
    child = await db_session.get(OrchestrationGoal, row.child_goal_id)
    child_run = await db_session.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == child.id))
    agent = await db_session.get(Agent, uuid.UUID(child.orchestrator_context["team"]["agent_ids"][0]))
    agent.adapter_type = "cli"
    task = Task(project_id=test_project.id, title="cli hours", status="ready", assigned_to=agent.id)
    db_session.add(task); await db_session.flush()
    await attach_claim_lineage(db_session, task, child_run, agent_id=agent.id)

    session = await SessionService().create(db_session, SessionCreate(
        agent_id=agent.id, task_id=task.id, project_id=test_project.id,
    ))
    assert session.metadata_["_run_config"] == {"timeout": 3600, "_roadmap_budget_enforced": True}
    assert await db_session.scalar(select(func.count(OrchestrationAuthorityDecision.id))) == 1


async def test_child_claim_uses_completed_action_lineage_not_editable_task_metadata(db_session, test_project):
    roadmap, parent, parent_run, _version = await accepted_goal_roadmap(
        db_session, test_project, goal_item("child", {"max_tokens": 400, "max_turns": 2, "max_hours": 1}),
        include_planner_in_team=True,
    )
    await roadmap.advance(db_session, parent, parent_run)
    row = await db_session.scalar(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == parent.id,
    ))
    child = await db_session.get(OrchestrationGoal, row.child_goal_id)
    child_run = await db_session.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == child.id))
    agent = await db_session.get(Agent, uuid.UUID(child.orchestrator_context["team"]["agent_ids"][0]))
    agent.capabilities = ["planning"]
    action = await roadmap.orchestration.execute_request_plan_action(
        db_session, child_run.id,
        {"action_type": "request_plan", "agent_id": str(agent.id), "work_function": "planning", "scope": "Plan child."},
        f"run:{child_run.id}:kind:immutable-budget-lineage",
    )
    task = await db_session.get(Task, action.target_id)
    task.metadata_ = {"orchestration": {"run_id": str(parent_run.id)}}

    session = await SessionService().create(db_session, SessionCreate(
        agent_id=agent.id, task_id=task.id, project_id=test_project.id, max_tokens=1000,
    ))
    assert session.metadata_["_run_config"]["max_tokens"] == 400

    task.metadata_ = {}
    assert await OrchestrationBudgetService().owning_roadmap_run(db_session, task) == child_run


async def test_unrelated_completed_action_does_not_grant_budget_ownership(db_session, test_project):
    roadmap, parent, run, _version = await accepted_goal_roadmap(
        db_session, test_project, goal_item("child", {"max_tokens": 400, "max_turns": 2, "max_hours": 1}),
        include_planner_in_team=True,
    )
    task = Task(project_id=test_project.id, title="unrelated action", status="ready")
    db_session.add(task); await db_session.flush()
    db_session.add(OrchestrationAction(
        run_id=run.id, idempotency_key=f"test:unrelated-action:{run.id}:{task.id}",
        action_type="complete_task", request={}, target_type="task", target_id=task.id, status="completed",
    ))
    await db_session.flush()
    assert await OrchestrationBudgetService().owning_roadmap_run(db_session, task) is None


async def test_competing_exact_cap_resume_refuses_then_recovers_after_terminal_usage(db_session, test_project):
    # The tasks carry orchestration metadata, so the resume authorization applies: the agent must be on the accepted team.
    roadmap, parent, parent_run, _version = await accepted_goal_roadmap(
        db_session, test_project, goal_item("child", {"max_tokens": 400, "max_turns": 5, "max_hours": 5}),
        include_planner_in_team=True,
    )
    await roadmap.advance(db_session, parent, parent_run)
    row = await db_session.scalar(select(OrchestrationRoadmapItem).where(OrchestrationRoadmapItem.goal_id == parent.id))
    child = await db_session.get(OrchestrationGoal, row.child_goal_id)
    child_run = await db_session.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == child.id))
    agent = await db_session.get(Agent, uuid.UUID(child.orchestrator_context["team"]["agent_ids"][0]))
    tasks = [Task(project_id=test_project.id, title=title, status="failed", assigned_to=agent.id,
        metadata_={"orchestration": {"run_id": str(child_run.id)}}) for title in ("first", "second")]
    db_session.add_all(tasks); await db_session.flush()
    for task in tasks:
        await attach_claim_lineage(db_session, task, child_run)
    sessions = [Session(task_id=task.id, agent_id=agent.id, project_id=test_project.id,
        adapter_type="api", status="failed", resumable=True,
        started_at=child_run.started_at, ended_at=child_run.started_at,
        metadata_={"token_count_in": 0, "token_count_out": 0,
                   "_run_config": {"max_tokens": 400, "timeout": 3600}}) for task in tasks]
    db_session.add_all(sessions); await db_session.flush()

    service = SessionService()
    first = await service.resume(db_session, sessions[0].id)
    assert first.metadata_["_run_config"]["max_tokens"] == 400
    with pytest.raises(SessionClaimAttention):
        await service.resume(db_session, sessions[1].id)
    with pytest.raises(SessionClaimAttention):
        await service.resume(db_session, sessions[1].id)
    assert sessions[1].status == "failed" and sessions[1].resumable is True

    first.status = "completed"
    first.started_at = child_run.started_at
    first.ended_at = child_run.started_at + timedelta(minutes=1)
    first.metadata_ = {"token_count_in": 50, "token_count_out": 50}
    recovered = await service.resume(db_session, sessions[1].id)
    assert recovered.status == "pending"
    assert recovered.metadata_["_run_config"]["max_tokens"] == 300


async def test_resume_reserves_prior_usage_plus_only_the_new_grant(db_session, test_project):
    _, _, _, _, row = await release(
        db_session, test_project, goal_item("child", {"max_tokens": 400, "max_turns": 5, "max_hours": 5}),
    )
    child_run = await db_session.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == row.child_goal_id))
    agent = _agent("resumed-cumulative-budget", ["implementation"])
    db_session.add(agent); await db_session.flush()
    task = Task(project_id=test_project.id, title="resumed", status="failed", assigned_to=agent.id)
    db_session.add(task); await db_session.flush()
    await attach_claim_lineage(db_session, task, child_run)
    session = Session(
        task_id=task.id, agent_id=agent.id, project_id=test_project.id, adapter_type="api",
        status="failed", resumable=True, started_at=child_run.started_at,
        ended_at=child_run.started_at + timedelta(hours=1),
        metadata_={"token_count_in": 60, "token_count_out": 40,
                   "_run_config": {"max_tokens": 400, "timeout": 18000,
                                   "_roadmap_budget_enforced": True}},
    )
    db_session.add(session); await db_session.flush()

    resumed = await SessionService().resume(db_session, session.id)
    assert resumed.metadata_["_run_config"]["max_tokens"] == 300
    commitments = await OrchestrationBudgetService().active_run_commitments(
        db_session, child_run, {"max_tokens", "max_turns", "max_hours"},
        {"max_tokens": "400", "max_turns": "5", "max_hours": "5"},
    )
    assert commitments == {"max_tokens": "400", "max_turns": "2", "max_hours": "2"}


async def test_active_cli_resume_charges_prior_overage_and_new_grant(db_session, test_project, monkeypatch):
    _, _, _, _, row = await release(
        db_session, test_project, goal_item("child", {"max_tokens": 800, "max_turns": 5, "max_hours": 5}),
    )
    child_run = await db_session.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == row.child_goal_id))
    agent = _agent("resumed-cli-grants", ["implementation"])
    db_session.add(agent); await db_session.flush()
    task = Task(project_id=test_project.id, title="resumed cli", status="failed", assigned_to=agent.id)
    db_session.add(task); await db_session.flush()
    await attach_claim_lineage(db_session, task, child_run)
    session = Session(
        task_id=task.id, agent_id=agent.id, project_id=test_project.id, adapter_type="cli",
        status="failed", resumable=True, started_at=child_run.started_at,
        ended_at=child_run.started_at + timedelta(hours=1),
        metadata_={
            "_roadmap_cli_budget_approval": {"decision_id": str(uuid.uuid4())},
            "_roadmap_cli_token_grants": [400], "token_usage_complete": False,
            "token_count_in": 450, "token_count_out": 0,
            "_run_config": {"max_tokens": 400, "timeout": 3600},
        },
    )
    db_session.add(session)
    await db_session.flush()

    async def approved(*_args, **_kwargs):
        return {"status": "approved", "decision_id": str(uuid.uuid4()), "unsupported_dimensions": ["max_tokens"]}

    monkeypatch.setattr(SessionService, "_cli_budget_authority", approved)
    monkeypatch.setattr(SessionService, "_schedule_dispatch_after_commit", lambda *_args: None)
    resumed = await SessionService().resume(db_session, session.id)
    assert resumed.metadata_["_roadmap_cli_token_grants"] == ["450", 350]

    commitments = await OrchestrationBudgetService().active_run_commitments(
        db_session, child_run, {"max_tokens", "max_turns", "max_hours"},
        {"max_tokens": "800", "max_turns": "5", "max_hours": "5"},
    )
    assert commitments == {"max_tokens": "800", "max_turns": "2", "max_hours": "2"}
    resumed.status = "completed"
    resumed.metadata_ = {**resumed.metadata_, "token_count_in": 350, "token_count_out": 0}
    assert await OrchestrationBudgetService().measured_run_spend(db_session, child_run, {"max_tokens"}) == {
        "max_tokens": "800"
    }
    resumed.status = "cancelled"
    known, complete = await OrchestrationBudgetService().known_run_spend(db_session, child_run, {"max_tokens"})
    assert (known, complete) == ({"max_tokens": "800"}, False)


async def test_repeated_resume_keeps_cumulative_usage_and_refuses_incomplete_telemetry(db_session, test_project):
    _, _, _, _, row = await release(
        db_session, test_project, goal_item("child", {"max_tokens": 400, "max_turns": 5, "max_hours": 5}),
    )
    child_run = await db_session.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == row.child_goal_id))
    agent = _agent("repeat-resume-budget", ["implementation"])
    db_session.add(agent); await db_session.flush()
    task = Task(project_id=test_project.id, title="repeated", status="failed", assigned_to=agent.id)
    db_session.add(task); await db_session.flush()
    await attach_claim_lineage(db_session, task, child_run)
    session = Session(
        task_id=task.id, agent_id=agent.id, project_id=test_project.id, adapter_type="api",
        status="failed", resumable=True, started_at=child_run.started_at,
        ended_at=child_run.started_at + timedelta(hours=1),
        metadata_={"token_count_in": 60, "token_count_out": 40,
                   "_run_config": {"max_tokens": 400, "timeout": 18000,
                                   "_roadmap_budget_enforced": True}},
    )
    db_session.add(session); await db_session.flush()
    service = SessionService()
    await service.resume(db_session, session.id)

    session.status = "failed"
    session.resumable = True
    session.started_at = child_run.started_at
    session.ended_at = child_run.started_at + timedelta(hours=2)
    session.metadata_ = {**session.metadata_, "token_count_in": 110, "token_count_out": 90,
                         "_roadmap_elapsed_seconds": "7200", "_roadmap_turn_count": 2}
    repeated = await service.resume(db_session, session.id)
    assert repeated.metadata_["_run_config"]["max_tokens"] == 200
    assert repeated.metadata_["_run_config"]["_roadmap_prior_usage"] == {
        "max_turns": "2", "max_tokens": "200", "max_hours": "2",
    }

    repeated.status = "failed"
    repeated.resumable = True
    repeated.metadata_ = {**repeated.metadata_, "token_usage_complete": False}
    with pytest.raises(SessionClaimAttention):
        await service.resume(db_session, repeated.id)
    assert repeated.status == "failed" and repeated.resumable is True


async def test_malformed_resume_elapsed_and_active_commitment_create_attention(db_session, test_project):
    _, _, _, _, row = await release(db_session, test_project)
    child_run = await db_session.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == row.child_goal_id))
    agent = _agent("malformed-budget", ["implementation"])
    db_session.add(agent); await db_session.flush()
    task = Task(project_id=test_project.id, title="malformed", status="failed", assigned_to=agent.id)
    db_session.add(task); await db_session.flush()
    await attach_claim_lineage(db_session, task, child_run)
    session = Session(
        task_id=task.id, agent_id=agent.id, project_id=test_project.id, adapter_type="api",
        status="failed", resumable=True, started_at=child_run.started_at,
        ended_at=child_run.started_at + timedelta(hours=1),
        metadata_={"token_count_in": 1, "token_count_out": 1, "_roadmap_elapsed_seconds": "NaN",
                   "_run_config": {"max_tokens": 400, "timeout": 3600,
                                   "_roadmap_budget_enforced": True}},
    )
    db_session.add(session); await db_session.flush()
    with pytest.raises(SessionClaimAttention):
        await SessionService().resume(db_session, session.id)


async def test_fractional_turn_cap_admits_only_one_active_claim(db_session, test_project):
    roadmap, parent, parent_run, _ = await accepted_goal_roadmap(
        db_session, test_project,
        goal_item("fractional", {"max_tokens": 800, "max_turns": 1.5, "max_hours": 10}),
        include_planner_in_team=True,
    )
    await roadmap.advance(db_session, parent, parent_run)
    row = await db_session.scalar(select(OrchestrationRoadmapItem).where(OrchestrationRoadmapItem.goal_id == parent.id))
    child_run = await db_session.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == row.child_goal_id))
    child = await db_session.get(OrchestrationGoal, row.child_goal_id)
    agent = await db_session.get(Agent, uuid.UUID(child.orchestrator_context["team"]["agent_ids"][0]))
    agent.capabilities = ["implementation"]
    tasks = [Task(project_id=test_project.id, title=f"turn-{index}", status="ready", assigned_to=agent.id)
             for index in range(2)]
    db_session.add_all(tasks); await db_session.flush()
    for task in tasks:
        await attach_claim_lineage(db_session, task, child_run, agent_id=agent.id)
    await SessionService().create(db_session, SessionCreate(
        agent_id=agent.id, task_id=tasks[0].id, project_id=test_project.id, max_tokens=100, timeout=10,
    ))
    with pytest.raises(SessionClaimAttention):
        await SessionService().create(db_session, SessionCreate(
            agent_id=agent.id, task_id=tasks[1].id, project_id=test_project.id, max_tokens=100, timeout=10,
        ))


async def test_invalid_active_commitment_is_claim_attention_not_negative_headroom(db_session, test_project):
    roadmap, parent, parent_run, _ = await accepted_goal_roadmap(
        db_session, test_project, goal_item(), include_planner_in_team=True,
    )
    await roadmap.advance(db_session, parent, parent_run)
    row = await db_session.scalar(select(OrchestrationRoadmapItem).where(OrchestrationRoadmapItem.goal_id == parent.id))
    child_run = await db_session.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == row.child_goal_id))
    child = await db_session.get(OrchestrationGoal, row.child_goal_id)
    agent = await db_session.get(Agent, uuid.UUID(child.orchestrator_context["team"]["agent_ids"][0]))
    agent.capabilities = ["implementation"]
    active_task = Task(project_id=test_project.id, title="bad-claim", status="in_progress", assigned_to=agent.id)
    next_task = Task(project_id=test_project.id, title="next-claim", status="ready", assigned_to=agent.id)
    db_session.add_all([active_task, next_task]); await db_session.flush()
    await attach_claim_lineage(db_session, active_task, child_run, agent_id=agent.id)
    await attach_claim_lineage(db_session, next_task, child_run, agent_id=agent.id)
    db_session.add(Session(
        task_id=active_task.id, agent_id=agent.id, project_id=test_project.id, adapter_type="api", status="running",
        metadata_={"_run_config": {"max_tokens": "NaN", "timeout": 3600}},
    ))
    await db_session.flush()
    with pytest.raises(SessionClaimAttention):
        await SessionService().create(db_session, SessionCreate(
            agent_id=agent.id, task_id=next_task.id, project_id=test_project.id,
        ))


async def test_outcome_runtime_stops_when_a_capped_dimension_is_exhausted(db_session, test_project):
    roadmap, _, _, _, row = await release(db_session, test_project)
    goal = await db_session.get(OrchestrationGoal, row.child_goal_id)
    child_run = await db_session.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == goal.id))
    child_run.budget_state = {"caps": {"max_tokens": 0, "max_turns": 1, "max_hours": 1}}
    assert await roadmap.orchestration._advance_authorized_execution(db_session, goal, child_run) == {"step": "budget_exhausted"}
    assert child_run.budget_state["status"] == "exceeded"


@pytest.mark.parametrize("value", [-1, "NaN", "Infinity"])
async def test_invalid_token_telemetry_is_rejected(db_session, test_project, value):
    _, parent, _, _, row = await release(db_session, test_project)
    child_run = await db_session.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == row.child_goal_id))
    agent = _agent("bad-telemetry", ["implementation"])
    db_session.add(agent); await db_session.flush()
    task = Task(project_id=test_project.id, title="bad", status="done",
        metadata_={"orchestration": {"run_id": str(child_run.id)}})
    db_session.add(task); await db_session.flush()
    await attach_claim_lineage(db_session, task, child_run)
    db_session.add(Session(task_id=task.id, agent_id=agent.id, project_id=test_project.id,
        adapter_type="test", status="completed", metadata_={"token_count_in": value, "token_count_out": 1}))
    with pytest.raises(BudgetMeasurementError):
        await OrchestrationBudgetService().measured_run_spend(db_session, child_run, {"max_tokens"})


# --- progressive child type resolution and inherited expansion snapshot ---

async def _row_counts(db, parent, run):
    return [
        await db.scalar(select(func.count()).select_from(model).where(condition))  # pylint: disable=not-callable
        for model, condition in [
            (OrchestrationGoal, OrchestrationGoal.parent_goal_id == parent.id),
            (OrchestrationRoadmapItem, OrchestrationRoadmapItem.goal_id == parent.id),
            (OrchestrationBudgetReservation, OrchestrationBudgetReservation.parent_goal_id == parent.id),
            (OrchestrationGate, (OrchestrationGate.run_id == run.id) & (OrchestrationGate.gate_type == "child_goal_completed")),
            (OrchestrationAction, (OrchestrationAction.run_id == run.id) & (OrchestrationAction.action_type == "release_roadmap_item")),
        ]
    ]


def _shape(goal_type, planning_mode):
    return GoalShapeDecision(
        goal_type=goal_type, goal_type_source="auto", goal_type_rationale="stub rationale",
        planning_mode=planning_mode, planning_rationale="stub planning rationale",
    )


@pytest.mark.parametrize("goal_type,planning_mode", [("outcome", "delivery"), ("roadmap", "roadmap")])
async def test_explicit_child_goal_type_is_preserved(db_session, test_project, goal_type, planning_mode):
    item = {**goal_item(), "goal_type": goal_type}
    roadmap, parent, run, version = await accepted_goal_roadmap(db_session, test_project, item)
    await roadmap.advance(db_session, parent, run)
    child = await db_session.scalar(select(OrchestrationGoal).where(OrchestrationGoal.parent_goal_id == parent.id))
    child_run = await db_session.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == child.id))
    assert (child.goal_type, child.goal_type_source, child.planning_mode) == (goal_type, "parent_explicit", planning_mode)
    assert f"declares goal_type {goal_type}" in child.goal_type_rationale and child.planning_rationale.strip()
    assert (child_run.goal_type_snapshot, child_run.planning_mode_snapshot) == (goal_type, planning_mode)
    assert child_run.goal_contract_fingerprint == compute_goal_contract_fingerprint(child)
    policy = child.orchestrator_context["expansion_policy"]
    assert policy["depth"] == 1 and policy["root_goal_id"] == str(parent.id)
    assert policy["max_goal_depth"] == 3 and policy["max_descendant_goals"] == 50
    assert policy["remaining_descendant_goals"] == 49
    assert "eligible_evidence" not in child.orchestrator_context


async def test_auto_child_is_classified_once_from_its_own_objective(db_session, test_project):
    calls = []

    async def classify(self, project_id, data):
        calls.append(data)
        return _shape("roadmap", "roadmap")

    item = {**goal_item(constraints={"language": "Python"}), "goal_type": "auto"}
    roadmap, parent, run, version = await accepted_goal_roadmap(db_session, test_project, item)
    with patch.object(OrchestrationGoalShapeService, "_classify", classify):
        await roadmap.advance(db_session, parent, run)
        await roadmap.advance(db_session, parent, run)
    assert len(calls) == 1
    assert (calls[0].objective, calls[0].success_criteria) == ("child", [{"key": "local"}])
    assert calls[0].constraints == {"region": "EU", "language": "Python"}
    child = await db_session.scalar(select(OrchestrationGoal).where(OrchestrationGoal.parent_goal_id == parent.id))
    assert (child.goal_type, child.goal_type_source, child.planning_mode) == ("roadmap", "parent_auto", "roadmap")
    assert child.goal_type_rationale == "stub rationale"


async def test_auto_child_provider_error_waits_without_persisting_anything(db_session, test_project):
    async def classify(self, project_id, data):
        raise GoalShapeResolutionError("provider_error", "boom")

    item = {**goal_item(), "goal_type": "auto"}
    roadmap, parent, run, version = await accepted_goal_roadmap(db_session, test_project, item)
    with patch.object(OrchestrationGoalShapeService, "_classify", classify):
        result = await roadmap.release_goal_item(
            db_session, parent, run, version, parse_roadmap_items([item], set(parent.budget["caps"]))[0],
        )
    assert result == {"step": "waiting", "reason": "child_shape_provider_error"}
    assert await _row_counts(db_session, parent, run) == [0, 0, 0, 0, 0]
    assert await db_session.scalar(select(OrchestrationRun).where(
        OrchestrationRun.id == uuid.uuid5(uuid.NAMESPACE_URL, f"rally:roadmap-child-run:{parent.id}:child"))) is None
    await roadmap.advance(db_session, parent, run)  # retried next tick with the stubbed classifier
    assert await _row_counts(db_session, parent, run) == [1, 1, 1, 1, 1]


async def test_auto_child_resolving_to_continuous_is_blocked_without_rows(db_session, test_project):
    async def classify(self, project_id, data):
        return _shape("continuous", "continuous_policy")

    item = {**goal_item(), "goal_type": "auto"}
    roadmap, parent, run, version = await accepted_goal_roadmap(db_session, test_project, item)
    with patch.object(OrchestrationGoalShapeService, "_classify", classify):
        result = await roadmap.release_goal_item(
            db_session, parent, run, version, parse_roadmap_items([item], set(parent.budget["caps"]))[0],
        )
    assert result == {"step": "needs_attention"}
    assert [b["reason"].split(":")[0] for b in run.active_blockers if b["kind"] == "child_contract"] == [
        "continuous_child_requires_continuous_root",
    ]
    assert await _row_counts(db_session, parent, run) == [0, 0, 0, 0, 0]


async def test_auto_child_classifier_runs_after_the_tick_releases_its_writer(
    db_session, test_project, concurrent_sessions, committed_roadmap_cleanup,
):
    item = {**goal_item(), "goal_type": "auto"}
    roadmap, parent, run, version = await accepted_goal_roadmap(db_session, test_project, item)
    parent_id, run_id = parent.id, run.id
    await db_session.commit()
    writer, reader = concurrent_sessions
    in_transaction = []

    async def classify(self, project_id, data):
        in_transaction.append(writer.in_transaction())
        return _shape("outcome", "delivery")

    with patch.object(OrchestrationGoalShapeService, "_classify", classify):
        result = await roadmap.orchestration.tick(writer, run_id)
    assert result["authorized_execution"] == {"step": "release_item", "item_key": "child", "unit_type": "goal"}
    assert in_transaction == [False]  # the tick committed before the model round trip
    child = await reader.scalar(select(OrchestrationGoal).where(OrchestrationGoal.parent_goal_id == parent_id))
    assert (child.goal_type, child.goal_type_source) == ("outcome", "parent_auto")
    assert await reader.scalar(select(func.count()).select_from(OrchestrationBudgetReservation).where(  # pylint: disable=not-callable
        OrchestrationBudgetReservation.parent_goal_id == parent_id)) == 1


async def test_rejected_auto_child_is_not_reclassified_until_a_new_version(db_session, test_project):
    calls = []

    async def classify(self, project_id, data):
        calls.append(data)
        return _shape("continuous", "continuous_policy")

    item = {**goal_item(), "goal_type": "auto"}
    roadmap, parent, run, version = await accepted_goal_roadmap(db_session, test_project, item)
    with patch.object(OrchestrationGoalShapeService, "_classify", classify):
        for _ in range(3):
            await roadmap.orchestration.tick(db_session, run.id)
    assert len(calls) == 1
    blocker, = [b for b in run.active_blockers if b["kind"] == "child_contract"]
    assert blocker["roadmap_version_id"] == str(version.id) and blocker["item_key"] == "child"
    assert await _row_counts(db_session, parent, run) == [0, 0, 0, 0, 0]
    run.active_blockers = [{**blocker, "roadmap_version_id": str(uuid.uuid4())}]  # as if a new version was accepted
    with patch.object(OrchestrationGoalShapeService, "_classify", classify):
        await roadmap.advance(db_session, parent, run)
    assert len(calls) == 2  # stale blocker cleared, item re-classified under the current version


async def test_continuous_child_item_is_rejected_with_reason():
    with pytest.raises(HTTPException) as exc:
        parse_roadmap_items([{**goal_item(), "goal_type": "continuous"}], {"max_tokens", "max_turns", "max_hours"})
    assert exc.value.status_code == 422 and "continuous_child_requires_continuous_root" in exc.value.detail


async def test_child_inherits_root_limits_and_eligible_evidence_from_accepted_dependency(db_session, test_project):
    artifact_id = uuid.uuid4()
    first = goal_item("first", {"max_tokens": 300, "max_turns": 1, "max_hours": 1})
    second = {**goal_item("second", {"max_tokens": 300, "max_turns": 1, "max_hours": 1}),
              "depends_on": ["first"], "goal_type": "outcome", "artifact_ids": [str(artifact_id)]}
    roadmap, parent, run, version = await accepted_goal_roadmap(db_session, test_project, [first, second])
    parent.orchestrator_context = {"expansion_policy": {"max_goal_depth": 2, "max_descendant_goals": 5, "depth": 0}}
    await roadmap.advance(db_session, parent, run)
    lineage = await db_session.scalar(select(OrchestrationRoadmapItem).where(OrchestrationRoadmapItem.item_key == "first"))
    gate = await db_session.get(OrchestrationGate, lineage.gate_id)
    gate.status = "accepted"
    evidence = OrchestrationEvidence(run_id=run.id, gate_id=gate.id, source_type="child_goal", verdict="accepted")
    db_session.add_all([evidence, OrchestrationEvidence(
        run_id=run.id, gate_id=gate.id, source_type="child_goal", verdict="rejected")])
    await db_session.flush()
    await roadmap.advance(db_session, parent, run)
    child = await db_session.scalar(select(OrchestrationGoal).where(
        OrchestrationGoal.parent_goal_id == parent.id, OrchestrationGoal.roadmap_item_key == "second"))
    context = child.orchestrator_context
    assert context["expansion_policy"]["max_goal_depth"] == 2
    assert context["expansion_policy"]["max_descendant_goals"] == 5
    assert context["expansion_policy"]["remaining_descendant_goals"] == 3
    assert context["eligible_evidence"] == {"artifact_ids": [str(artifact_id)], "evidence_ids": [str(evidence.id)]}


# --- atomic depth / descendant-count rejection ---

def _policy(**limits):
    return {"expansion_policy": {"depth": 0, **limits}}


async def test_expansion_limit_error_messages():
    limits = {"max_goal_depth": 3, "max_descendant_goals": 50}
    assert expansion_limit_error(2, 49, 1, limits) is None
    assert expansion_limit_error(3, 0, 0, limits) is None
    assert "child depth 4 exceeds max_goal_depth 3" in expansion_limit_error(3, 0, 1, limits)
    assert "51 descendant goals exceed max_descendant_goals 50" in expansion_limit_error(1, 49, 2, limits)


async def test_release_beyond_depth_limit_is_rejected_with_no_rows(db_session, test_project):
    item = {**goal_item(), "mutates_shared_state": True}
    roadmap, parent, run, version = await accepted_goal_roadmap(db_session, test_project, item)
    parent.orchestrator_context = _policy(max_goal_depth=0)
    decisions = await db_session.scalar(select(func.count()).select_from(OrchestrationAuthorityDecision))  # pylint: disable=not-callable
    result = await roadmap.release_goal_item(
        db_session, parent, run, version, parse_roadmap_items([item], set(parent.budget["caps"]))[0],
    )
    assert result == {"step": "needs_attention"}
    assert await _row_counts(db_session, parent, run) == [0, 0, 0, 0, 0]
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAuthorityDecision)) == decisions  # pylint: disable=not-callable
    blocker, = [b for b in run.active_blockers if b["kind"] == "expansion_limit"]
    assert blocker["item_key"] == "child" and blocker["roadmap_version_id"] == str(version.id)
    assert blocker["required_decision"] == "request_roadmap_replan" and "max_goal_depth 0" in blocker["reason"]


async def test_tick_rejects_over_depth_mutable_goal_item_before_any_authority_decision(db_session, test_project):
    item = {**goal_item(), "mutates_shared_state": True}
    roadmap, parent, run, version = await accepted_goal_roadmap(db_session, test_project, item)
    parent.orchestrator_context = _policy(max_goal_depth=0)
    decisions = await db_session.scalar(select(func.count()).select_from(OrchestrationAuthorityDecision))  # pylint: disable=not-callable
    await roadmap.orchestration.tick(db_session, run.id)
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAuthorityDecision)) == decisions  # pylint: disable=not-callable
    assert await _row_counts(db_session, parent, run) == [0, 0, 0, 0, 0]
    blocker, = [b for b in run.active_blockers if b["kind"] == "expansion_limit"]
    assert blocker["item_key"] == "child" and "max_goal_depth 0" in blocker["reason"]


async def test_release_beyond_descendant_limit_is_rejected_after_the_first_child(db_session, test_project):
    small = {"max_tokens": 300, "max_turns": 1, "max_hours": 1}
    roadmap, parent, run, version = await accepted_goal_roadmap(
        db_session, test_project, [goal_item("one", small), goal_item("two", small)],
    )
    parent.orchestrator_context = _policy(max_descendant_goals=1)
    await roadmap.advance(db_session, parent, run)
    await roadmap.advance(db_session, parent, run)
    assert await _row_counts(db_session, parent, run) == [1, 1, 1, 1, 1]
    blocker, = [b for b in run.active_blockers if b["kind"] == "expansion_limit"]
    assert blocker["item_key"] == "two" and "max_descendant_goals 1" in blocker["reason"]


async def test_depth_is_measured_from_the_finite_root_through_released_children(db_session, test_project):
    item = {**goal_item(), "goal_type": "roadmap"}
    roadmap, parent, run, version = await accepted_goal_roadmap(db_session, test_project, item)
    await roadmap.advance(db_session, parent, run)
    child = await db_session.scalar(select(OrchestrationGoal).where(OrchestrationGoal.parent_goal_id == parent.id))
    parent.orchestrator_context = _policy(max_goal_depth=1)
    assert "child depth 2 exceeds max_goal_depth 1" in await roadmap._expansion_limit_error(db_session, child, 1)
    parent.orchestrator_context = _policy(max_goal_depth=2)
    assert await roadmap._expansion_limit_error(db_session, child, 1) is None


async def test_literal_three_deep_goal_tree_allows_depth_three_and_rejects_depth_four(db_session, test_project):
    item = {**goal_item(), "goal_type": "roadmap"}
    roadmap, root, run, version = await accepted_goal_roadmap(db_session, test_project, item)
    await roadmap.advance(db_session, root, run)  # root -> child (depth 1, real release)
    chain = [root, await db_session.scalar(select(OrchestrationGoal).where(OrchestrationGoal.parent_goal_id == root.id))]
    for depth, goal_type in ((2, "roadmap"), (3, "outcome")):
        node = OrchestrationGoal(
            project_id=test_project.id, parent_goal_id=chain[-1].id, objective=f"d{depth}", original_request=f"d{depth}",
            success_criteria=[], constraints={}, budget={}, goal_type=goal_type, status="active",
            continuous_origin_key=f"d{depth}", parent_contract_snapshot={}, goal_delta={},
        )
        db_session.add(node); await db_session.flush()
        chain.append(node)
    assert root.orchestrator_context.get("expansion_policy", {}).get("max_goal_depth") in (None, 3)
    # parents at depth 0,1,2 may add a child (child depth 1,2,3); the depth-3 goal may not (child depth 4)
    for parent in chain[:3]:
        assert await roadmap._expansion_limit_error(db_session, parent, 1) is None
    assert "child depth 4 exceeds max_goal_depth 3" in await roadmap._expansion_limit_error(db_session, chain[3], 1)


async def test_stale_expansion_limit_blocker_is_cleared_for_a_new_version(db_session, test_project):
    roadmap, parent, run, version = await accepted_goal_roadmap(db_session, test_project, goal_item())
    run.active_blockers = [{"kind": "expansion_limit", "item_key": "child", "roadmap_version_id": str(uuid.uuid4()),
                            "reason": "expansion_limit: stale", "required_decision": "request_roadmap_replan"}]
    await roadmap.advance(db_session, parent, run)
    assert not [b for b in run.active_blockers if b["kind"] == "expansion_limit"]
    assert await _row_counts(db_session, parent, run) == [1, 1, 1, 1, 1]


# --- nested budget: a child reservation carries its whole subtree ---

NESTED_ITEM = {**goal_item("child", {"max_tokens": 800, "max_turns": 5, "max_hours": 5}), "goal_type": "roadmap"}


async def _nested_child(db, project, *grand):
    """Release Roadmap child C and give it roadmap-item grandchildren: (settled_tokens | None for active, allocation)."""
    _, parent, run, _, row = await release(db, project, NESTED_ITEM)
    child = await db.get(OrchestrationGoal, row.child_goal_id)
    assert child.goal_type == "roadmap"
    child_run = await db.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == child.id))
    plan = Artifact(project_id=project.id, name="c-plan", artifact_type="plan", status="draft")
    db.add(plan); await db.flush()
    child_version = OrchestrationRoadmapVersion(
        goal_id=child.id, run_id=child_run.id, version=1, plan_artifact_id=plan.id,
        snapshot={"schema_version": 1, "items": []}, fingerprint="c" * 64, approval_reference={},
    )
    db.add(child_version); await db.flush()
    for index, (settled, tokens, complete) in enumerate(grand):
        grandchild = OrchestrationGoal(
            project_id=project.id, parent_goal_id=child.id, objective=f"g{index}", original_request=f"g{index}",
            success_criteria=[], constraints={}, budget={}, goal_type="outcome", status="active",
            roadmap_version_id=child_version.id, roadmap_item_key=f"g{index}", parent_contract_snapshot={}, goal_delta={},
        )
        gate = OrchestrationGate(run_id=child_run.id, success_criterion_key=f"roadmap_item:g{index}",
                                 gate_type="child_goal_completed", required_evidence={})
        db.add_all([grandchild, gate]); await db.flush()
        lineage = OrchestrationRoadmapItem(
            goal_id=child.id, first_version_id=child_version.id, item_key=f"g{index}", unit_type="goal",
            item_snapshot={}, child_goal_id=grandchild.id, gate_id=gate.id,
        )
        db.add(lineage); await db.flush()
        reservation = OrchestrationBudgetReservation(
            parent_goal_id=child.id, child_goal_id=grandchild.id, roadmap_item_id=lineage.id,
            allocation={"max_tokens": str(tokens), "max_turns": "1", "max_hours": "1"},
        )
        if settled is not None:
            reservation.status, reservation.settlement_reason = "settled", "completed"
            reservation.settled_spend = {"max_tokens": str(settled), "max_turns": "1", "max_hours": "1"}
            reservation.measurement_complete, reservation.settled_at = complete, datetime.now(timezone.utc)
        db.add(reservation)
    await db.flush()
    reservation = await db.scalar(select(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.child_goal_id == child.id))
    return parent, child_run, reservation


def _direct_spend(tokens):
    async def measured(self, db, run, dimensions, **_kwargs):
        return {key: (str(tokens) if key == "max_tokens" else "1") for key in dimensions}

    async def known(self, db, run, dimensions):
        return await measured(self, db, run, dimensions), True
    return patch.multiple(OrchestrationBudgetService, measured_run_spend=measured, known_run_spend=known)


async def test_settle_folds_the_subtree_into_the_child_reservation(db_session, test_project):
    parent, child_run, reservation = await _nested_child(db_session, test_project, (250, 300, True))
    before = await OrchestrationBudgetService().remaining(db_session, parent)
    with _direct_spend(100):  # C's own run spends 100, G settled 250 -> 350 charged to P once
        await OrchestrationBudgetService().settle_child(db_session, reservation, child_run, reason="completed")
    assert reservation.settled_spend["max_tokens"] == "350"
    assert reservation.measurement_complete is True
    after = await OrchestrationBudgetService().remaining(db_session, parent)
    assert before["remaining"]["max_tokens"] == "200" and after["remaining"]["max_tokens"] == "650"


async def test_active_reservation_hides_grandchild_spend_from_the_parent(db_session, test_project):
    parent, _, _ = await _nested_child(db_session, test_project, (250, 300, True), (None, 200, True))
    summary = await OrchestrationBudgetService().remaining(db_session, parent)
    assert summary["active_reservations"]["max_tokens"] == "800" and summary["remaining"]["max_tokens"] == "200"


async def test_cancelled_child_counts_active_grandchild_allocation(db_session, test_project):
    parent, child_run, reservation = await _nested_child(db_session, test_project, (250, 300, True), (None, 200, True))
    with _direct_spend(100):
        await OrchestrationBudgetService().settle_child(
            db_session, reservation, child_run, reason="cancelled", conservative_cancel=False)
    assert reservation.settled_spend["max_tokens"] == "550"  # 100 direct + 250 settled + 200 still reserved


async def test_conservative_cancel_and_overage_apply_to_the_subtree_total(db_session, test_project):
    parent, child_run, reservation = await _nested_child(db_session, test_project, (700, 700, True))
    with _direct_spend(200), pytest.raises(HTTPException) as exc:  # 900 > 800 allocation
        await OrchestrationBudgetService().settle_child(db_session, reservation, child_run, reason="completed")
    assert exc.value.detail == "Child exceeded reserved max_tokens"
    with _direct_spend(200):
        await OrchestrationBudgetService().settle_child(db_session, reservation, child_run, reason="cancelled")
    assert reservation.settled_spend["max_tokens"] == "900"  # conservative max(actual, allocation) keeps the larger


async def test_incomplete_grandchild_measurement_marks_the_child_incomplete(db_session, test_project):
    parent, child_run, reservation = await _nested_child(db_session, test_project, (50, 300, False))
    with _direct_spend(10):
        await OrchestrationBudgetService().settle_child(db_session, reservation, child_run, reason="completed")
    assert reservation.settled_spend["max_tokens"] == "60" and reservation.measurement_complete is False


async def test_outcome_child_settlement_is_unchanged(db_session, test_project):
    parent, child_run, reservation = await _nested_child(db_session, test_project)
    with _direct_spend(120):
        await OrchestrationBudgetService().settle_child(db_session, reservation, child_run, reason="completed")
    assert reservation.settled_spend == {"max_tokens": "120", "max_turns": "1", "max_hours": "1"}


async def test_nested_roadmap_cli_cap_authority_keys_by_its_own_version_and_item(db_session, test_project):
    """A nested Roadmap's own capped task item asks on its goal/run/version, as the claim side resolves it."""
    _, parent, _, _, row = await release(db_session, test_project, NESTED_ITEM)
    child = await db_session.get(OrchestrationGoal, row.child_goal_id)
    child_run = await db_session.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == child.id))
    plan = Artifact(project_id=test_project.id, name="c-plan", artifact_type="plan", status="draft")
    agent = _agent("nested-cli", ["implementation"])
    agent.adapter_type = "cli"
    db_session.add_all([plan, agent]); await db_session.flush()
    version = OrchestrationRoadmapVersion(
        goal_id=child.id, run_id=child_run.id, version=1, plan_artifact_id=plan.id,
        snapshot={"schema_version": 1, "items": []}, fingerprint="d" * 64, approval_reference={},
    )
    db_session.add(version); await db_session.flush()
    child.budget = {"caps": {"max_tokens": 800, "max_turns": 5, "max_hours": 5}}
    context = {"roadmap": {"roadmap_version_id": str(version.id), "roadmap_item_key": "own-task"}}

    result = await OrchestrationService().roadmap_pre_release_authority(db_session, child_run.id, agent, context)

    assert result == {"status": "pending"}
    decision = await db_session.scalar(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.decision_key.like("roadmap_budget_adapter:%:cli:max_tokens"),
        OrchestrationAuthorityDecision.status == "pending",
    ))
    assert (decision.goal_id, decision.run_id) == (child.id, child_run.id)
    assert decision.decision_key.startswith(f"roadmap_budget_adapter:{version.id}:own-task:{agent.id}:")


async def test_nested_roadmap_first_version_freezes_the_parents_workspace_policy(db_session, test_project):
    _, parent, _, _, row = await release(db_session, test_project, NESTED_ITEM)
    child = await db_session.get(OrchestrationGoal, row.child_goal_id)
    child_run = await db_session.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == child.id))
    assert child.orchestrator_context["workspace_policy"] == {"mode": "isolated"}
    test_project.config = {"workspace_policy": {"mode": "changed-later"}}
    planner = _agent("nested-planner", ["planning"])
    db_session.add(planner); await db_session.flush()
    items = [goal_item("g1", {"max_tokens": 200, "max_turns": 1, "max_hours": 1})]
    artifact = Artifact(project_id=test_project.id, name="c-plan", artifact_type="plan", status="draft",
                        created_by_agent=planner.id, metadata_={"plan_items": items})
    db_session.add(artifact); await db_session.flush()
    service = OrchestrationService()
    normalized = [x.model_dump(mode="json") for x in parse_roadmap_items(items, {"max_tokens", "max_turns", "max_hours"})]
    decision = OrchestrationAuthorityDecision(
        goal_id=child.id, run_id=child_run.id, authority="human", title="approve", question="approve",
        decision_key=f"roadmap_plan:{service._accepted_plan_fingerprint(normalized)}", status="answered",
        selected_option="approve", decided_by_user_id=child.manager_user_id,
    )
    db_session.add(decision); await db_session.flush()

    version = await OrchestrationRoadmapService(service).accept_version(
        db_session, child, child_run, artifact, {"kind": "authority_decision", "id": str(decision.id)},
    )

    assert version.snapshot["workspace_policy"] == {"mode": "isolated"}


async def test_nested_roadmap_planning_task_keeps_the_lineage_it_was_released_under(db_session, test_project):
    """C's own planning claim carries P's version/item (inherited context), so it is authorized on P."""
    _, parent, parent_run, _ = await accepted_goal_roadmap(
        db_session, test_project, NESTED_ITEM, include_planner_in_team=True)
    roadmap = OrchestrationRoadmapService(OrchestrationService())
    await roadmap.advance(db_session, parent, parent_run)
    row = await db_session.scalar(select(OrchestrationRoadmapItem).where(OrchestrationRoadmapItem.goal_id == parent.id))
    child = await db_session.get(OrchestrationGoal, row.child_goal_id)
    child_run = await db_session.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == child.id))
    agent = await db_session.get(Agent, uuid.UUID(child.orchestrator_context["team"]["agent_ids"][0]))
    agent.adapter_type = "cli"
    context = {"roadmap": {"roadmap_version_id": str(child.roadmap_version_id), "roadmap_item_key": row.item_key}}

    assert (await roadmap.orchestration.roadmap_pre_release_authority(
        db_session, child_run.id, agent, context))["status"] == "pending"
    decision = await db_session.scalar(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.decision_key.like("roadmap_budget_adapter:%:cli:max_tokens")))
    assert decision.goal_id == parent.id and decision.run_id == parent_run.id
    assert decision.decision_key.startswith(f"roadmap_budget_adapter:{child.roadmap_version_id}:{row.item_key}:")


async def _released_nested_roadmap_with_cli_planner(db, project):
    roadmap, parent, parent_run, _ = await accepted_goal_roadmap(db, project, NESTED_ITEM, include_planner_in_team=True)
    await roadmap.advance(db, parent, parent_run)
    row = await db.scalar(select(OrchestrationRoadmapItem).where(OrchestrationRoadmapItem.goal_id == parent.id))
    child = await db.get(OrchestrationGoal, row.child_goal_id)
    child_run = await db.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == child.id))
    agent = await db.get(Agent, uuid.UUID(child.orchestrator_context["team"]["agent_ids"][0]))
    agent.adapter_type = "cli"
    return roadmap, parent, child, child_run, agent


async def _answer(db, goal, like, user_id):
    decision = await db.scalar(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.goal_id == goal.id, OrchestrationAuthorityDecision.status == "pending",
        OrchestrationAuthorityDecision.decision_key.like(like)))
    await OrchestrationAuthorityDecisionService().answer_decision(
        db, decision, selected_option="approve", decided_by_user_id=user_id)
    return decision


async def test_nested_roadmap_cli_planning_claim_is_authorized_on_the_inherited_lineage(db_session, test_project):
    """Release-time approval and the claim agree: C's planning task runs under the lineage C was released under."""
    roadmap, parent, child, child_run, agent = await _released_nested_roadmap_with_cli_planner(db_session, test_project)
    service = roadmap.orchestration
    request = {"action_type": "request_plan", "agent_id": str(agent.id), "work_function": "planning",
               "scope": "Plan the nested Roadmap."}
    key = f"run:{child_run.id}:kind:request_plan:nested-cli"

    with pytest.raises(HTTPException, match="budget_wait"):
        await service.execute_request_plan_action(db_session, child_run.id, request, key)
    decision = await _answer(db_session, parent, "roadmap_budget_adapter:%:cli:max_tokens", parent.manager_user_id)
    assert (decision.goal_id, decision.decision_key.split(":")[1]) == (parent.id, str(child.roadmap_version_id))
    action = await service.execute_request_plan_action(db_session, child_run.id, request, key)
    task = await db_session.get(Task, action.target_id)

    session = await SessionService().create(db_session, SessionCreate(
        agent_id=agent.id, task_id=task.id, project_id=test_project.id))
    assert session.metadata_["_roadmap_cli_budget_approval"]["decision_id"] == str(decision.id)


async def test_nested_roadmap_own_cli_task_item_claim_keys_on_its_own_lineage(db_session, test_project):
    from tests.test_orchestration_roadmap_task_items import roadmap_task

    roadmap, parent, child, child_run, agent = await _released_nested_roadmap_with_cli_planner(db_session, test_project)
    service = roadmap.orchestration
    child.budget = {"caps": {"max_tokens": 800, "max_turns": 5, "max_hours": 5}}
    from sqlalchemy import update
    from huddleroom.models.orchestration import OrchestrationRun, compute_goal_contract_fingerprint
    await db_session.execute(update(OrchestrationRun).where(OrchestrationRun.id == child_run.id).values(goal_contract_fingerprint=compute_goal_contract_fingerprint(child))); await db_session.refresh(child_run)
    items = [{**roadmap_task("own-task"), "agent_id": str(agent.id)}]
    artifact = Artifact(project_id=test_project.id, name="c-plan", artifact_type="plan", status="draft",
                        created_by_agent=agent.id, metadata_={"plan_items": items})
    db_session.add(artifact); await db_session.flush()
    normalized = [x.model_dump(mode="json") for x in parse_roadmap_items(items, {"max_tokens", "max_turns", "max_hours"})]
    approval = OrchestrationAuthorityDecision(
        goal_id=child.id, run_id=child_run.id, authority="human", title="approve", question="approve",
        decision_key=f"roadmap_plan:{service._accepted_plan_fingerprint(normalized)}", status="answered",
        selected_option="approve", decided_by_user_id=child.manager_user_id)
    db_session.add(approval); await db_session.flush()
    version = await roadmap.accept_version(
        db_session, child, child_run, artifact, {"kind": "authority_decision", "id": str(approval.id)})
    child_run.plan_state = {**child_run.plan_state, "status": "accepted", "accepted_artifact_id": str(artifact.id),
                            "roadmap_version_id": str(version.id), "roadmap_version": version.version,
                            "accepted_plan_fingerprint": version.fingerprint}

    await roadmap.advance(db_session, child, child_run)
    decision = await _answer(db_session, child, "roadmap_budget_adapter:%:own-task:%:cli:max_tokens", child.manager_user_id)
    assert decision.decision_key.startswith(f"roadmap_budget_adapter:{version.id}:own-task:")
    await roadmap.advance(db_session, child, child_run)
    row = await db_session.scalar(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == child.id, OrchestrationRoadmapItem.item_key == "own-task"))
    session = await SessionService().create(db_session, SessionCreate(
        agent_id=agent.id, task_id=row.task_id, project_id=test_project.id))
    assert session.metadata_["_roadmap_cli_budget_approval"]["decision_id"] == str(decision.id)


async def test_grandchild_cli_cap_authority_lands_on_the_nested_roadmap(db_session, test_project):
    roadmap, parent, child, child_run, agent = await _released_nested_roadmap_with_cli_planner(db_session, test_project)
    service = roadmap.orchestration
    items = [goal_item("grand", {"max_tokens": 300, "max_turns": 2, "max_hours": 1})]
    artifact = Artifact(project_id=test_project.id, name="c-plan", artifact_type="plan", status="draft",
                        created_by_agent=agent.id, metadata_={"plan_items": items})
    db_session.add(artifact); await db_session.flush()
    normalized = [x.model_dump(mode="json") for x in parse_roadmap_items(items, {"max_tokens", "max_turns", "max_hours"})]
    approval = OrchestrationAuthorityDecision(
        goal_id=child.id, run_id=child_run.id, authority="human", title="approve", question="approve",
        decision_key=f"roadmap_plan:{service._accepted_plan_fingerprint(normalized)}", status="answered",
        selected_option="approve", decided_by_user_id=child.manager_user_id)
    db_session.add(approval); await db_session.flush()
    version = await roadmap.accept_version(
        db_session, child, child_run, artifact, {"kind": "authority_decision", "id": str(approval.id)})
    child_run.plan_state = {**child_run.plan_state, "status": "accepted", "accepted_artifact_id": str(artifact.id),
                            "roadmap_version_id": str(version.id), "roadmap_version": version.version,
                            "accepted_plan_fingerprint": version.fingerprint}
    await roadmap.advance(db_session, child, child_run)
    grand = await db_session.scalar(select(OrchestrationGoal).where(OrchestrationGoal.parent_goal_id == child.id))
    grand_run = await db_session.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == grand.id))
    context = {"roadmap": {"roadmap_version_id": str(version.id), "roadmap_item_key": "grand"}}

    assert (await service.roadmap_pre_release_authority(db_session, grand_run.id, agent, context))["status"] == "pending"
    decision = await db_session.scalar(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.decision_key.like("roadmap_budget_adapter:%:grand:%:cli:max_tokens")))
    assert (decision.goal_id, decision.run_id) == (child.id, child_run.id)
    assert decision.decision_key.startswith(f"roadmap_budget_adapter:{version.id}:grand:")
