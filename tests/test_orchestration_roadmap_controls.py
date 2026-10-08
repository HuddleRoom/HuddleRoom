from datetime import timedelta

import pytest
from fastapi import HTTPException
from sqlalchemy import delete, func, select

from huddleroom.models.orchestration import (
    OrchestrationAction,
    OrchestrationBudgetReservation,
    OrchestrationGoal,
    OrchestrationRoadmapItem,
    OrchestrationRoadmapVersion,
    OrchestrationRun,
    OrchestrationGate, OrchestrationEvidence,
)
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.services.orchestration_budget_service import BudgetMeasurementError, OrchestrationBudgetService
from huddleroom.services.orchestration_roadmap_service import OrchestrationRoadmapService
from huddleroom.services.orchestration_supersession import OrchestrationSupersessionService
from huddleroom.routers.orchestration_goals import _detail, get_goal, pause_goal
from tests.test_orchestration_roadmap_e2e import (
    _approved_roadmap, _ready_roadmap, _snapshot, _active_child_work, _active_session, _late_output,
)
from tests.test_orchestration_roadmap_integration import (
    _child_actors, _complete_child, _local_decisions, child_item, release,
)
from tests.test_orchestration_roadmap_task_items import roadmap_task
from tests.test_orchestration_runtime_e2e import _agent, _authorized_run
from huddleroom.services.orchestration_service import OrchestrationService


@pytest.fixture(autouse=True)
def _manual_task_start(monkeypatch):
    # These tests drive the release-then-manual-run flow; auto-start is covered in test_orchestration_task_autostart.py.
    async def _noop(self, db, goal, run):
        return 0
    monkeypatch.setattr(OrchestrationService, "_start_released_tasks", _noop)


pytestmark = pytest.mark.asyncio


async def test_start_authorizes_ready_roadmap_but_still_rejects_continuous(db_session, test_project):
    service, roadmap, roadmap_run, _planner = await _ready_roadmap(db_session, test_project)
    assert roadmap_run.phase == "authorized"
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
        OrchestrationAction.run_id == roadmap_run.id,
        OrchestrationAction.action_type == "authorize_execution",
    )) == 1

    service, continuous, continuous_run = await _authorized_run(db_session, test_project)
    continuous.goal_type = "continuous"
    continuous_run.phase = "ready"
    with pytest.raises(HTTPException) as exc:
        await service.start_run(db_session, test_project.id, continuous.id, actor="human:test")
    assert exc.value.detail["conflict"] == "goal_not_runnable"


async def test_pause_stops_new_release_while_active_child_may_finish(
    db_session, test_project, monkeypatch, safe_effectiveness_review_continue,
):
    planner, producer, verifier, summarizer = await _child_actors(db_session)
    builder = _agent("pause-docs", ["implementation"])
    db_session.add(builder); await db_session.flush()
    child = child_item("rollout")
    docs = {**roadmap_task("docs", depends_on=["rollout"]), "agent_id": str(builder.id)}
    service, parent, run, _ = await _approved_roadmap(db_session, test_project, [child, docs])
    roadmap = OrchestrationRoadmapService(service)
    _local_decisions(monkeypatch, roadmap.orchestration, lambda _ctx: {"action_type": "noop", "reason": "paused"})
    await service.tick(db_session, run.id)
    row = await db_session.scalar(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == parent.id, OrchestrationRoadmapItem.item_key == "rollout"))
    child = await db_session.get(OrchestrationGoal, row.child_goal_id)
    child_run = await db_session.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == child.id))

    await roadmap.orchestration.pause_goal(db_session, test_project.id, parent.id)
    assert child.status == "active" and child_run.status == "running"
    await _complete_child(db_session, test_project, roadmap, row, planner, producer, verifier, summarizer)
    before = await _snapshot(db_session, (OrchestrationBudgetReservation, OrchestrationRoadmapItem, OrchestrationEvidence))
    await service.tick(db_session, run.id)
    assert await _snapshot(db_session, (OrchestrationBudgetReservation, OrchestrationRoadmapItem, OrchestrationEvidence)) == before
    assert await db_session.scalar(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == parent.id, OrchestrationRoadmapItem.item_key == "docs")) is None


async def test_resume_settles_finished_child_and_continues(
    db_session, test_project, monkeypatch, safe_effectiveness_review_continue,
):
    planner, producer, verifier, summarizer = await _child_actors(db_session)
    builder = _agent("resume-docs", ["implementation"])
    db_session.add(builder); await db_session.flush()
    child_item_spec = child_item("rollout")
    docs = {**roadmap_task("docs", depends_on=["rollout"]), "agent_id": str(builder.id)}
    service, parent, run, _ = await _approved_roadmap(db_session, test_project, [child_item_spec, docs])
    roadmap = OrchestrationRoadmapService(service)
    _local_decisions(monkeypatch, roadmap.orchestration, lambda _ctx: {"action_type": "noop", "reason": "resume"})
    await service.tick(db_session, run.id)
    row = await db_session.scalar(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == parent.id, OrchestrationRoadmapItem.item_key == "rollout"))
    child = await db_session.get(OrchestrationGoal, row.child_goal_id)
    await roadmap.orchestration.pause_goal(db_session, test_project.id, parent.id)
    await _complete_child(db_session, test_project, roadmap, row, planner, producer, verifier, summarizer)
    await roadmap.orchestration.resume_goal(db_session, test_project.id, parent.id)
    first = (await service.tick(db_session, run.id))["authorized_execution"]
    reservation = await db_session.scalar(select(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.roadmap_item_id == row.id
    ))
    assert reservation.status == "settled"
    assert first["step"] == "settle_children"
    assert await db_session.scalar(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == parent.id, OrchestrationRoadmapItem.item_key == "docs")) is None
    assert (await db_session.get(OrchestrationGate, row.gate_id)).status == "accepted"
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationEvidence).where(
        OrchestrationEvidence.gate_id == row.gate_id, OrchestrationEvidence.source_type == "child_goal")) == 1
    second = (await service.tick(db_session, run.id))["authorized_execution"]
    assert second["item_key"] == "docs"


async def _cancellable_parent(db_session, test_project):
    planner, producer, _verifier, _summarizer = await _child_actors(db_session)
    service, parent, run, _ = await _approved_roadmap(db_session, test_project, [
        {**roadmap_task("parent-work", mutates_shared_state=False), "agent_id": str(producer.id)},
        child_item("child"),
    ])
    await service.tick(db_session, run.id)
    await service.tick(db_session, run.id)
    rows = {row.item_key: row for row in await db_session.scalars(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == parent.id))}
    parent_task = await db_session.get(Task, rows["parent-work"].task_id)
    parent_session = await _active_session(db_session, test_project.id, parent_task)
    child_run, child_task, child_session = await _active_child_work(
        db_session, test_project, service, rows["child"], planner, producer)
    return service, parent, run, rows["child"], child_run, (parent_task, child_task), (parent_session, child_session)


async def test_cancel_cascades_to_unfinished_descendants_and_settles_once(db_session, test_project):
    service, parent, run, row, child_run, tasks, sessions = await _cancellable_parent(db_session, test_project)
    child = await db_session.get(OrchestrationGoal, row.child_goal_id)

    await service.cancel_goal(db_session, test_project.id, parent.id)
    reservation = await db_session.scalar(select(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.roadmap_item_id == row.id
    ))
    assert (parent.status, child.status, child_run.status, reservation.status) == (
        "cancelled", "cancelled", "cancelled", "settled"
    )
    assert reservation.measurement_complete is False
    assert reservation.settled_spend == reservation.allocation
    assert [task.status for task in tasks] == ["cancelled", "cancelled"]
    assert [session.status for session in sessions] == ["cancelled", "cancelled"]
    assert await db_session.scalar(select(func.count()).select_from(Session).where(
        Session.task_id.in_([task.id for task in tasks]), Session.status.in_(["running", "pending"]))) == 0
    before = await _snapshot(db_session, (OrchestrationBudgetReservation, OrchestrationAction))
    with pytest.raises(HTTPException) as exc:
        await service.cancel_goal(db_session, test_project.id, parent.id)
    assert exc.value.status_code == 409 and exc.value.detail == "Cannot cancel goal in status 'cancelled'"
    await service.tick(db_session, child_run.id)
    await service.tick(db_session, run.id)
    assert await _snapshot(db_session, (OrchestrationBudgetReservation, OrchestrationAction)) == before
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.roadmap_item_id == row.id,
        OrchestrationBudgetReservation.status == "settled")) == 1


async def test_late_child_output_after_parent_cancel_accepts_no_gate(db_session, test_project):
    service, parent, run, row, child_run, tasks, sessions = await _cancellable_parent(db_session, test_project)
    await service.cancel_goal(db_session, test_project.id, parent.id)
    before = await _snapshot(db_session, (OrchestrationGate, OrchestrationEvidence, OrchestrationBudgetReservation))
    await _late_output(db_session, test_project.id, tasks[1], sessions[1])
    for _ in range(2):
        await service.tick(db_session, child_run.id)
        await service.tick(db_session, run.id)
    assert await _snapshot(db_session, (OrchestrationGate, OrchestrationEvidence, OrchestrationBudgetReservation)) == before
    assert (await db_session.get(OrchestrationGate, row.gate_id)).status != "accepted"
    assert tasks[1].status == sessions[1].status == "cancelled"


@pytest.mark.parametrize("blocker", ["descendant", "reservation"])
async def test_supersession_rejects_unfinished_descendant_or_active_reservation(db_session, test_project, blocker):
    roadmap, parent, run, _, row = await release(db_session, test_project)
    # Supersession is intentionally legal only before Start.  Recreate that
    # otherwise-eligible control state so each Roadmap-specific predicate,
    # rather than the earlier authorized-phase guard, is the rejection reason.
    run.phase = "ready"
    # The accepted-plan task is unrelated to the Roadmap safety predicate
    # under test; close it so `_require_ready` reaches descendants/reservations.
    for task in await db_session.scalars(select(Task).where(
        Task.metadata_["orchestration"]["run_id"].as_string() == str(run.id)
    )):
        task.status = "done"
    if blocker == "descendant":
        await db_session.execute(delete(OrchestrationBudgetReservation).where(
            OrchestrationBudgetReservation.roadmap_item_id == row.id
        ))
    else:
        child = await db_session.get(OrchestrationGoal, row.child_goal_id)
        child.status = "cancelled"
    models = (OrchestrationGoal, OrchestrationRun, OrchestrationRoadmapVersion,
              OrchestrationRoadmapItem, OrchestrationBudgetReservation, OrchestrationAction)
    before = await _snapshot(db_session, models)
    with pytest.raises(HTTPException) as exc:
        await OrchestrationSupersessionService().supersede(
            db_session, test_project.id, parent.id, new_goal_type="outcome", actor="human:test"
        )
    assert exc.value.detail["conflict"] == "supersession_not_ready"
    assert exc.value.detail["message"] == (
        "unfinished descendants" if blocker == "descendant" else "active child reservations"
    )
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationGoal).where(
        OrchestrationGoal.supersedes_goal_id == parent.id)) == 0
    assert await _snapshot(db_session, models) == before


async def test_reset_rejects_goal_after_roadmap_lineage_exists(db_session, test_project):
    roadmap, parent, _, _, row = await release(db_session, test_project)
    with pytest.raises(HTTPException) as exc:
        await roadmap.orchestration.reset_goal(db_session, test_project.id, parent.id)
    assert exc.value.status_code == 409
    assert await db_session.scalar(select(OrchestrationRoadmapVersion).where(
        OrchestrationRoadmapVersion.goal_id == parent.id
    )) is not None
    assert await db_session.scalar(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.id == row.id
    )) is not None


async def test_reset_allows_fresh_roadmap_without_immutable_lineage(db_session, test_project):
    service, roadmap, run = await _authorized_run(db_session, test_project)
    roadmap.goal_type = "roadmap"
    await db_session.flush()

    _, replacement = await service.reset_goal(db_session, test_project.id, roadmap.id)

    assert replacement.id != run.id
    assert await db_session.scalar(select(OrchestrationRoadmapVersion.id).where(
        OrchestrationRoadmapVersion.goal_id == roadmap.id,
    )) is None
    assert await db_session.scalar(select(OrchestrationRoadmapItem.id).where(
        OrchestrationRoadmapItem.goal_id == roadmap.id,
    )) is None


async def _fresh_goal_with_evidence(db_session, test_project, key):
    goal = OrchestrationGoal(project_id=test_project.id, objective=f"fresh-{key}")
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id)
    db_session.add(run)
    await db_session.flush()
    return goal, run


@pytest.mark.parametrize("lineage", ["parent", "item", "reservation"])
async def test_reset_rejects_each_immutable_child_lineage_without_mutation(
    db_session, test_project, lineage,
):
    service, parent, _run, row, child_run, tasks, sessions = await _cancellable_parent(
        db_session, test_project,
    )
    child = await db_session.get(OrchestrationGoal, row.child_goal_id)
    reservation = await db_session.scalar(select(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.roadmap_item_id == row.id,
    ))
    assert child is not None and reservation is not None
    target, target_run = (child, child_run)
    if lineage != "parent":
        target, target_run = await _fresh_goal_with_evidence(db_session, test_project, lineage)
        if lineage == "item":
            row.child_goal_id = target.id
        else:
            reservation.child_goal_id = target.id
    gate = OrchestrationGate(
        run_id=target_run.id, success_criterion_key=f"reset-{lineage}", gate_type="manual",
    )
    db_session.add(gate)
    await db_session.flush()
    db_session.add(OrchestrationEvidence(
        run_id=target_run.id, gate_id=gate.id, source_type="test", verdict="accepted",
    ))
    await db_session.flush()

    assert target_run.status == "running"
    assert [task.status for task in tasks] == ["in_progress", "in_progress"]
    assert [session.status for session in sessions] == ["running", "running"]
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationEvidence).where(
        OrchestrationEvidence.run_id == target_run.id,
    )) > 0
    if lineage == "parent":
        assert target.parent_goal_id == parent.id
    elif lineage == "item":
        assert target.parent_goal_id is None and row.child_goal_id == target.id
        assert reservation.child_goal_id != target.id
    else:
        assert target.parent_goal_id is None and row.child_goal_id != target.id
        assert reservation.child_goal_id == target.id and reservation.status == "active"
    before = await _snapshot(db_session, (
        OrchestrationGoal, OrchestrationRun, OrchestrationAction, OrchestrationGate,
        OrchestrationEvidence, OrchestrationRoadmapItem, OrchestrationRoadmapVersion,
        OrchestrationBudgetReservation, Task, Session,
    ))

    for _ in range(2):
        with pytest.raises(HTTPException) as exc:
            await service.reset_goal(db_session, test_project.id, target.id)
        assert exc.value.status_code == 409
        assert await _snapshot(db_session, (
            OrchestrationGoal, OrchestrationRun, OrchestrationAction, OrchestrationGate,
            OrchestrationEvidence, OrchestrationRoadmapItem, OrchestrationRoadmapVersion,
            OrchestrationBudgetReservation, Task, Session,
        )) == before


async def test_goal_detail_projects_version_items_children_and_budget_summary(db_session, test_project):
    roadmap, parent, _, _, _ = await release(db_session, test_project)
    detail = await _detail(db_session, parent, await roadmap.orchestration.get_run_for_goal(db_session, test_project.id, parent.id))
    assert detail.roadmap_version and detail.roadmap_items and detail.children
    assert detail.budget_summary and set(detail.budget_summary.model_dump()) == {
        "caps", "direct_spend", "settled_child_spend", "active_reservations", "active_commitments", "remaining"
    }
    assert "snapshot" not in detail.roadmap_version.model_dump()
    public = detail.model_dump(mode="json")
    forbidden = {"snapshot", "item_snapshot", "parent_contract_snapshot", "goal_delta", "reservation"}

    def walk(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(public)


async def test_goal_detail_leaves_roadmap_only_fields_null_for_outcome(db_session, test_project):
    service, goal, run = await _authorized_run(db_session, test_project)
    detail = await _detail(db_session, goal, run)
    assert detail.roadmap_version is None
    assert detail.roadmap_items is None
    assert detail.children is None
    assert detail.budget_summary is None


@pytest.mark.parametrize("work_function", ["planning", "implementation", "validation"])
async def test_parent_missing_telemetry_blocks_tick_then_recovers_without_spending(
    db_session, test_project, safe_effectiveness_review_continue, work_function,
):
    """A repaired parent session must re-open Roadmap admission, regardless of its role."""
    service, goal, run, _ = await _approved_roadmap(db_session, test_project, [child_item("build")])
    agent = _agent(f"missing-{work_function}", [work_function])
    db_session.add(agent)
    await db_session.flush()
    task = Task(
        project_id=test_project.id, title=work_function, assigned_to=agent.id, status="done",
        metadata_={"orchestration": {"run_id": str(run.id), "work_function": work_function}},
    )
    db_session.add(task)
    await db_session.flush()
    db_session.add(OrchestrationAction(
        run_id=run.id, idempotency_key=f"test:parent-measurement:{run.id}:{task.id}",
        action_type="create_delegation_task", request={}, target_type="task", target_id=task.id,
        status="completed",
    ))
    await db_session.flush()
    session = Session(
        task_id=task.id, agent_id=agent.id, project_id=test_project.id, adapter_type="test", status="completed",
        started_at=run.started_at, ended_at=run.started_at + timedelta(minutes=1), metadata_={},
    )
    db_session.add(session)
    await db_session.flush()
    with pytest.raises(BudgetMeasurementError, match="Missing max_tokens measurement"):
        await OrchestrationBudgetService().remaining(db_session, goal)

    get_detail = await get_goal(project_id=test_project.id, goal_id=goal.id, db=db_session)
    assert get_detail.budget_summary is None
    assert get_detail.supervision.budget == {}
    assert get_detail.supervision.condition == "needs_attention"
    assert any(item["scope"] == f"parent:{goal.id}" for item in get_detail.run.active_blockers)

    blocked = await service.tick(db_session, run.id)
    assert blocked["authorized_execution"] == {"step": "waiting", "reason": "needs_attention"}
    assert any(item["kind"] == "budget_measurement" and "item_key" not in item for item in run.active_blockers)
    assert await db_session.scalar(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == goal.id
    )) is None
    detail = await _detail(db_session, goal, run)
    assert detail.budget_summary is None
    assert any(item["scope"] == f"parent:{goal.id}" for item in detail.run.active_blockers)

    session.metadata_ = {"token_count_in": 1, "token_count_out": 2}
    recovered = await service.tick(db_session, run.id)
    assert recovered["authorized_execution"] == {"step": "release_item", "item_key": "build", "unit_type": "goal"}
    assert not any(item["kind"] == "budget_measurement" and "item_key" not in item for item in run.active_blockers)
    session.metadata_ = {}
    paused = await pause_goal(project_id=test_project.id, goal_id=goal.id, db=db_session)
    assert paused.budget_summary is None
    assert any(item["scope"] == f"parent:{goal.id}" for item in paused.run.active_blockers)


async def test_measurement_blockers_keep_parent_and_child_scopes_on_replay(db_session, test_project):
    service, goal, run, _ = await _approved_roadmap(db_session, test_project, [child_item("build")])
    parent = {"kind": "budget_measurement", "scope": f"parent:{goal.id}", "session_id": "parent"}
    child = {"kind": "budget_measurement", "scope": "roadmap_item:build", "session_id": "child"}
    unrelated = {"kind": "staging_boundary", "item_key": "other"}
    for blocker in (parent, child, unrelated, parent, child):
        service._upsert_active_blocker(run, blocker)
    assert run.active_blockers == [unrelated, parent, child]
