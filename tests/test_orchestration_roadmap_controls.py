from datetime import timedelta
from contextvars import ContextVar
import json

import pytest
from fastapi import HTTPException
from sqlalchemy import delete, func, null, select, update
from sqlalchemy.ext.asyncio import AsyncSession

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
from huddleroom.dependencies import _ANON_USER
from huddleroom.schemas.session import SessionCreate
from huddleroom.services.session_service import SessionService
from huddleroom.services.task_service import TaskService
from huddleroom.services.session_sync import sync_task_from_session
from huddleroom.services.event_bus import emit_event
from huddleroom.workers.session_tasks import _require_runnable_session, mark_attempt_effect_started

_owned_roadmap_setup = ContextVar("owned_roadmap_controls_setup", default=None)


@pytest.fixture
async def db_session(test_engine):
    """SPR230/SPR231 native control tests own their committed setup."""
    async with AsyncSession(test_engine, expire_on_commit=False) as db:
        yield db
        await db.rollback()
        # Committed Roadmap fixtures contain intentional RESTRICT lineage cycles.
        # Break those only after assertions, on this isolated test engine.
        await db.execute(update(OrchestrationGoal).values(parent_goal_id=None, roadmap_version_id=None,
            roadmap_item_key=None, parent_contract_snapshot=null(), goal_delta=null(), continuous_origin_key=None))
        await db.commit()


@pytest.fixture(autouse=True)
def _owned_tick_setup(db_session, monkeypatch):
    original = OrchestrationService.tick

    async def tick(self, db, *args, **kwargs):
        root = db.sync_session.get_transaction()
        if ((db is db_session or db is _owned_roadmap_setup.get())
                and not db.in_nested_transaction()
                and (root is None or root.origin.name == "AUTOBEGIN")):
            # Publish only this module's lexical setup before the real provider-capable tick.
            # Explicit/nested transactions and HTTP/snapshot sessions retain their own boundaries.
            await db.flush()
            await db.commit()
        return await original(self, db, *args, **kwargs)

    monkeypatch.setattr(OrchestrationService, "tick", tick)


@pytest.fixture(autouse=True)
def _manual_task_start(monkeypatch):
    # These tests drive the release-then-manual-run flow; auto-start is covered in test_orchestration_task_autostart.py.
    async def _noop(self, db, goal, run):
        return 0
    monkeypatch.setattr(OrchestrationService, "_start_released_tasks", _noop)


pytestmark = pytest.mark.asyncio


async def _complete_child_across_parent_pause(db, project, roadmap, row, actors, monkeypatch):
    """SPR230 admit every child step before Pause; retain its final in-flight result."""
    from tests import test_orchestration_roadmap_integration as integration

    original = integration._record_measured_completion
    paused = False

    class RetainedFinalResult(Exception):
        pass

    async def complete(db, project_id, run, task, agent_id, output):
        nonlocal paused
        is_summary = "summary" in json.loads(output)
        if is_summary:
            assert not paused
            await db.commit()
            _, session_id = await TaskService().run(db, project_id, task.id)
            await db.commit()
            session = await db.get(Session, session_id)
            runner_id = session.runner_task_id
            assert await _require_runnable_session(db, session_id, runner_id)
            assert await mark_attempt_effect_started(db, session_id, runner_id)
            await roadmap.orchestration.pause_goal(db, project.id, row.goal_id)
            paused = True
            session = await SessionService().update_status(db, session_id, "completed", output=output,
                metadata={**session.metadata_, "token_count_in": 1, "token_count_out": 1})
            session.started_at = session.ended_at - timedelta(minutes=1)
            await emit_event(db, project_id, "session.completed", {
                "session_id": str(session_id), "task_id": str(task.id),
            })
            await sync_task_from_session(db, session)
            await db.commit()
            raise RetainedFinalResult(session.id)
        return await original(db, project_id, run, task, agent_id, output)

    with monkeypatch.context() as patch:
        patch.setattr(integration, "_record_measured_completion", complete)
        with pytest.raises(RetainedFinalResult) as retained:
            await _complete_child(db, project, roadmap, row, *actors)
    assert paused
    session = await db.get(Session, retained.value.args[0])
    assert session.status == "completed" and "summary" in json.loads(session.output)
    child_run = await db.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == row.child_goal_id))
    before = await _snapshot(db, (OrchestrationAction, Task, Session))
    with pytest.raises(HTTPException, match="authorization_denied: run_not_active") as denied:
        await SessionService().create(db, SessionCreate(
            project_id=project.id, task_id=session.task_id, agent_id=session.agent_id))
    assert denied.value.status_code == 409
    assert await _snapshot(db, (OrchestrationAction, Task, Session)) == before
    return child_run, session


async def test_start_authorizes_ready_roadmap_but_still_rejects_continuous(db_session, test_project):
    service, roadmap, roadmap_run, _planner = await _ready_roadmap(db_session, test_project)
    assert roadmap_run.phase == "authorized"
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
        OrchestrationAction.run_id == roadmap_run.id,
        OrchestrationAction.action_type == "authorize_execution",
    )) == 1

    service, continuous, continuous_run = await _authorized_run(db_session, test_project)
    continuous.goal_type = "continuous"
    continuous.planning_mode = "continuous_policy"
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

    assert child.status == "active" and child_run.status == "running"
    await _complete_child_across_parent_pause(db_session, test_project, roadmap, row,
        (planner, producer, verifier, summarizer), monkeypatch)
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
    child_run, retained = await _complete_child_across_parent_pause(db_session, test_project, roadmap, row,
        (planner, producer, verifier, summarizer), monkeypatch)
    retained_id, retained_task_id, retained_output = retained.id, retained.task_id, retained.output
    await roadmap.orchestration.resume_goal(db_session, test_project.id, parent.id)
    await service.tick(db_session, child_run.id)
    await db_session.refresh(child)
    assert child.status == "completed"
    once = await _snapshot(db_session, (OrchestrationAction, OrchestrationEvidence, Task, Session))
    await service.tick(db_session, child_run.id)
    assert await _snapshot(db_session, (OrchestrationAction, OrchestrationEvidence, Task, Session)) == once
    assert list(await db_session.scalars(select(Session.id).where(Session.task_id == retained_task_id))) == [retained_id]
    assert (await db_session.get(Session, retained_id)).output == retained_output
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
    roadmap.planning_mode = "roadmap"
    await db_session.flush()

    original_run_id = run.id
    _, replacement = await service.reset_goal(db_session, test_project.id, roadmap.id)

    assert replacement.id != original_run_id
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
    assert get_detail.supervision.condition == "unknown"
    assert get_detail.roadmap_budget_status == "unavailable"
    assert not any(item.get("scope") == f"parent:{goal.id}" for item in get_detail.run.active_blockers)

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
    blockers_before_pause = list(run.active_blockers)
    paused = await pause_goal(project_id=test_project.id, goal_id=goal.id, user=_ANON_USER, db=db_session)
    assert paused.budget_summary is None
    assert paused.run.active_blockers == blockers_before_pause


async def test_measurement_blockers_keep_parent_and_child_scopes_on_replay(db_session, test_project):
    service, goal, run, _ = await _approved_roadmap(db_session, test_project, [child_item("build")])
    parent = {"kind": "budget_measurement", "scope": f"parent:{goal.id}", "session_id": "parent"}
    child = {"kind": "budget_measurement", "scope": "roadmap_item:build", "session_id": "child"}
    unrelated = {"kind": "staging_boundary", "item_key": "other"}
    for blocker in (parent, child, unrelated, parent, child):
        service._upsert_active_blocker(run, blocker)
    assert run.active_blockers == [unrelated, parent, child]


@pytest.mark.parametrize("measured,existing", [(False, False), (False, True), (True, True)])
async def test_repeated_roadmap_get_is_pure_across_committed_sessions(test_engine, monkeypatch, tmp_path, measured, existing):
    """Commit request sessions so autoflush writes cannot hide behind fixture rollback."""
    from copy import deepcopy
    from httpx import AsyncClient, ASGITransport
    from sqlalchemy import event
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
    from huddleroom.main import create_app
    from huddleroom.database import get_db
    from huddleroom.dependencies import get_current_user, _ANON_USER
    from huddleroom.models.project import Project
    from huddleroom.models.base import Base
    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    async with factory() as db:
        project = Project(name="Read purity", workspace_path=str(tmp_path))
        db.add(project); await db.flush()
        setup_token = _owned_roadmap_setup.set(db)
        try:
            _, goal, run, _ = await _approved_roadmap(db, project, [child_item("build")])
        finally:
            _owned_roadmap_setup.reset(setup_token)
        agent = _agent("measurement", ["implementation"])
        db.add(agent); await db.flush()
        task = Task(project_id=project.id, title="Measured work", assigned_to=agent.id, status="done",
                    metadata_={"orchestration": {"run_id": str(run.id)}})
        db.add(task); await db.flush()
        db.add(OrchestrationAction(run_id=run.id, idempotency_key="measurement-source",
            action_type="create_delegation_task", request={}, target_type="task", target_id=task.id, status="completed"))
        db.add(Session(task_id=task.id, agent_id=agent.id, project_id=project.id, adapter_type="test", status="completed",
            started_at=run.started_at, ended_at=run.started_at + timedelta(minutes=1),
            metadata_={"token_count_in": 1, "token_count_out": 2} if measured else {}))
        if existing:
            run.active_blockers = [{"kind": "budget_measurement", "scope": f"parent:{goal.id}"},
                {"kind": "budget_measurement", "scope": f"claim:{goal.id}"},
                {"kind": "budget_measurement", "scope": "parent:unrelated"}]
        ids = project.id, goal.id
        await db.commit()
    async def snapshot():
        async with factory() as db:
            return {table.name: [dict(row) for row in (await db.execute(select(table))).mappings()]
                    for table in Base.metadata.sorted_tables}
    before = await snapshot()
    app = create_app()
    async def request_db():
        async with factory() as db:
            yield db
            await db.commit()
    async def anon():
        return _ANON_USER
    app.dependency_overrides[get_db] = request_db
    app.dependency_overrides[get_current_user] = anon
    dml, flushes = [], []
    def sql(conn, cursor, statement, parameters, context, many):
        if statement.lstrip().split()[0].upper() in {"INSERT", "UPDATE", "DELETE", "REPLACE"}:
            dml.append(statement)
    from sqlalchemy.orm import Session as SyncSession
    def flushed(session, context, instances):
        flushes.append(True)
    async def forbidden(*args, **kwargs):
        raise AssertionError("GET must not call provider or execution reconciliation")
    monkeypatch.setattr(OrchestrationRoadmapService, "remaining_or_block", forbidden)
    monkeypatch.setattr(OrchestrationService, "request_llm_decision", forbidden)
    event.listen(test_engine.sync_engine, "before_cursor_execute", sql)
    event.listen(SyncSession, "before_flush", flushed)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            for _ in range(2):
                response = await client.get(f"/api/v1/projects/{ids[0]}/orchestration/goals/{ids[1]}")
                assert response.status_code == 200, response.text
                if not measured:
                    assert response.json()["roadmap_budget_status"] == "unavailable"
                assert await snapshot() == before
        assert dml == []
        assert flushes == []
    finally:
        event.remove(test_engine.sync_engine, "before_cursor_execute", sql)
        event.remove(SyncSession, "before_flush", flushed)
