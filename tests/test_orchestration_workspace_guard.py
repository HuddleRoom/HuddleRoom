import asyncio
import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy import event, func, select

from huddleroom.models.event_log import EventLog
from huddleroom.models.orchestration import (
    OrchestrationAction,
    OrchestrationAgentSuggestion,
    OrchestrationEvidence,
    OrchestrationGate,
    OrchestrationGoal,
    OrchestrationRun,
)
from huddleroom.models.orchestration_process import OrchestrationProcessRun
from huddleroom.models.project import Project
from huddleroom.routers.orchestration_processes import start_process
from huddleroom.schemas.orchestration import OrchestrationGoalCreate, OrchestrationProcessStartRequest
from huddleroom.schemas.project import ProjectUpdate
from huddleroom.services.orchestration_process_service import OrchestrationProcessService
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.project_service import ProjectService


def _assert_not_runnable(exc_info: pytest.ExceptionInfo[HTTPException]) -> None:
    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == {"code": "project_not_runnable", "reason": "workspace_unset"}


@pytest.mark.asyncio
async def test_goal_create_preserves_missing_project_404(db_session):
    with pytest.raises(HTTPException) as exc_info:
        await OrchestrationService().create_goal(
            db_session,
            uuid.uuid4(),
            OrchestrationGoalCreate(objective="missing project"),
            created_by_user_id=None,
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Project not found"


@pytest.mark.asyncio
async def test_goal_create_succeeds_for_runnable_project(db_session, test_project, tmp_path):
    test_project.workspace_path = str(tmp_path.resolve())
    await db_session.flush()

    goal, run = await OrchestrationService().create_goal(
        db_session,
        test_project.id,
        OrchestrationGoalCreate(objective="runnable goal"),
        created_by_user_id=None,
    )

    assert goal.project_id == test_project.id
    assert run.goal_id == goal.id


@pytest.mark.asyncio
async def test_goal_create_rejects_unrunnable_project_before_goal_or_run_mutation(db_session, legacy_project):
    with pytest.raises(HTTPException) as exc_info:
        await OrchestrationService().create_goal(
            db_session,
            legacy_project.id,
            OrchestrationGoalCreate(objective="guarded goal"),
            created_by_user_id=None,
        )

    _assert_not_runnable(exc_info)
    assert not (await db_session.execute(select(OrchestrationGoal))).scalars().all()
    assert not (await db_session.execute(select(OrchestrationRun))).scalars().all()


@pytest.mark.asyncio
async def test_resume_and_tick_reject_unrunnable_project_without_mutating_goal_or_run(db_session, legacy_project):
    goal = OrchestrationGoal(project_id=legacy_project.id, objective="guarded goal", status="paused")
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id, status="paused")
    db_session.add(run)
    await db_session.flush()

    service = OrchestrationService()
    with pytest.raises(HTTPException) as resume_error:
        await service.resume_goal(db_session, legacy_project.id, goal.id)
    _assert_not_runnable(resume_error)

    run.status = "running"
    await db_session.flush()
    with pytest.raises(HTTPException) as tick_error:
        await service.tick(db_session, run.id)
    _assert_not_runnable(tick_error)

    await db_session.refresh(goal)
    await db_session.refresh(run)
    assert goal.status == "paused"
    assert run.status == "running"

    goals, _ = await service.list_goals(db_session, legacy_project.id)
    assert [item.id for item in goals] == [goal.id]
    cancelled_goal, cancelled_run = await service.cancel_goal(db_session, legacy_project.id, goal.id)
    assert cancelled_goal.status == "cancelled"
    assert cancelled_run is not None and cancelled_run.status == "cancelled"


@pytest.mark.asyncio
async def test_process_start_and_delegation_reject_unrunnable_project_without_dispatch(db_session, legacy_project):
    goal = OrchestrationGoal(project_id=legacy_project.id, objective="guarded goal")
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id)
    db_session.add(run)
    await db_session.flush()

    with pytest.raises(HTTPException) as process_error:
        await start_process(
            legacy_project.id,
            goal.id,
            "goal_definition",
            OrchestrationProcessStartRequest(reason="test"),
            db=db_session,
        )
    _assert_not_runnable(process_error)

    with pytest.raises(HTTPException) as delegation_error:
        await OrchestrationService().execute_create_delegation_task_action(
            db_session,
            run.id,
            request={},
            idempotency_key="workspace-guard",
        )
    _assert_not_runnable(delegation_error)

    assert not (await db_session.execute(select(OrchestrationProcessRun))).scalars().all()
    assert not (await db_session.execute(select(OrchestrationAction))).scalars().all()


@pytest.mark.asyncio
async def test_heavier_weight_override_rejects_before_process_restart(db_session, legacy_project):
    goal = OrchestrationGoal(project_id=legacy_project.id, objective="tiny change", weight="trivial")
    db_session.add(goal)
    await db_session.flush()
    first = await OrchestrationProcessService().start_process(
        db_session,
        goal.id,
        process_type="goal_definition",
        trigger_reason="test",
    )
    first.status = "completed"
    await db_session.flush()

    with pytest.raises(HTTPException) as exc_info:
        await OrchestrationService().override_goal_weight(
            db_session,
            legacy_project.id,
            goal.id,
            weight="substantial",
            reason="force full pass",
            user_id=None,
        )

    _assert_not_runnable(exc_info)
    await db_session.refresh(goal)
    assert goal.weight == "trivial"
    assert goal.weight_overridden_by is None
    processes = list((await db_session.execute(select(OrchestrationProcessRun))).scalars())
    assert [(process.id, process.status, process.superseded_by_id) for process in processes] == [
        (first.id, "completed", None)
    ]


@pytest.mark.asyncio
async def test_goal_create_serializes_workspace_update(concurrent_sessions, tmp_path, monkeypatch):
    claim_db, update_db = concurrent_sessions
    old_workspace = tmp_path / "old"
    new_workspace = tmp_path / "new"
    old_workspace.mkdir()
    new_workspace.mkdir()
    project = Project(name="Project", workspace_path=str(old_workspace.resolve()), config={})
    claim_db.add(project)
    await claim_db.commit()

    claim_validated = asyncio.Event()
    release_claim = asyncio.Event()
    update_boundary_reached = asyncio.Event()
    original_require = ProjectService.require_runnable_project

    async def hold_after_validation(service, db, project_id):
        workspace = await original_require(service, db, project_id)
        claim_validated.set()
        await release_claim.wait()
        return workspace

    def signal_update_boundary(_conn, _cursor, statement, _parameters, _context, _executemany):
        if statement.startswith("UPDATE projects SET id=projects.id"):
            update_boundary_reached.set()

    monkeypatch.setattr(ProjectService, "require_runnable_project", hold_after_validation)
    engine = update_db.bind
    claim = update = None
    primary_error = None
    listener_installed = False
    try:
        claim = asyncio.create_task(
            OrchestrationService().create_goal(
                claim_db,
                project.id,
                OrchestrationGoalCreate(objective="serialized goal"),
                created_by_user_id=None,
            )
        )
        await asyncio.wait_for(claim_validated.wait(), timeout=1)

        event.listen(engine.sync_engine, "before_cursor_execute", signal_update_boundary)
        listener_installed = True
        update = asyncio.create_task(
            ProjectService().update(
                update_db,
                project.id,
                ProjectUpdate(workspace_path=str(new_workspace.resolve())),
            )
        )
        await asyncio.wait_for(update_boundary_reached.wait(), timeout=1)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(update), timeout=0.1)

        release_claim.set()
        await asyncio.wait_for(claim, timeout=1)
        await claim_db.commit()
        with pytest.raises(HTTPException) as update_error:
            await asyncio.wait_for(update, timeout=1)
        assert update_error.value.status_code == 409
        assert update_error.value.detail == "Cannot update workspace while project work is active"
        assert "database is locked" not in str(update_error.value)
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        release_claim.set()
        for task in (claim, update):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    if primary_error is None:
                        raise
            elif task is not None:
                try:
                    task.exception()
                except asyncio.CancelledError:
                    pass
        if listener_installed:
            event.remove(engine.sync_engine, "before_cursor_execute", signal_update_boundary)


@pytest.mark.asyncio
async def test_final_summary_weak_fit_rejects_unrunnable_project_without_mutation(
    db_session, test_project, tmp_path
):
    from tests.conftest import complete_baseline_processes

    test_project.workspace_path = str(tmp_path.resolve())
    await db_session.flush()
    service = OrchestrationService()
    goal, run = await service.create_goal(
        db_session,
        test_project.id,
        OrchestrationGoalCreate(objective="summarize completed work"),
        created_by_user_id=None,
    )
    run.baseline_authorized = True
    await complete_baseline_processes(db_session, goal, run)
    gate = OrchestrationGate(
        run_id=run.id,
        success_criterion_key="done",
        gate_type="work_completed",
        required_evidence={"required_source_types": ["task"], "min_count": 1},
        status="accepted",
    )
    db_session.add(gate)
    await db_session.flush()
    db_session.add(
        OrchestrationEvidence(
            run_id=run.id,
            gate_id=gate.id,
            source_type="task",
            source_id=uuid.uuid4(),
            verdict="accepted",
            evidence_metadata={"fixture": "workspace_guard"},
        )
    )
    test_project.workspace_path = None
    await db_session.flush()
    before = (
        await db_session.scalar(select(func.count(OrchestrationAction.id))),
        await db_session.scalar(select(func.count(OrchestrationAgentSuggestion.id))),
        await db_session.scalar(select(func.count(EventLog.id))),
    )

    with pytest.raises(HTTPException) as exc_info:
        await service.execute_request_final_summary_action(
            db_session,
            run.id,
            {"action_type": "request_final_summary", "work_function": "summarization"},
            f"run:{run.id}:kind:request_final_summary",
        )

    _assert_not_runnable(exc_info)
    after = (
        await db_session.scalar(select(func.count(OrchestrationAction.id))),
        await db_session.scalar(select(func.count(OrchestrationAgentSuggestion.id))),
        await db_session.scalar(select(func.count(EventLog.id))),
    )
    assert after == before
