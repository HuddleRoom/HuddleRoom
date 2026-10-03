import asyncio

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import select

from huddleroom.dependencies import _ANON_USER
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGoal
from huddleroom.models.project import Project
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.routers.orchestration_goals import supersede_goal
from huddleroom.schemas.orchestration import OrchestrationGoalCreate, OrchestrationSupersedeRequest
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.orchestration_supersession import OrchestrationSupersessionService


pytestmark = pytest.mark.asyncio


async def test_supersede_request_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        OrchestrationSupersedeRequest.model_validate({"unexpected": True})


async def _goal(db_session, test_project, *, phase="baseline"):
    goal, run = await OrchestrationService().create_goal(
        db_session,
        test_project.id,
        OrchestrationGoalCreate(
            objective="Replace this goal safely.",
            success_criteria=[{"key": "done", "description": "Done"}],
            budget={"cap_usd": 100, "nested": {"preserve": True}},
        ),
        created_by_user_id=None,
    )
    run.phase = phase
    await db_session.flush()
    return goal, run


async def test_supersede_baseline_cancels_and_creates_fresh_budget_matched_replacement(
    db_session, test_project
):
    goal, run = await _goal(db_session, test_project)

    replacement = await OrchestrationSupersessionService().supersede(
        db_session, test_project.id, goal.id, new_goal_type="outcome", actor="human:test"
    )

    assert replacement.supersedes_goal_id == goal.id
    assert replacement.budget == goal.budget
    assert replacement.goal_type == "outcome"
    replacement_run = await OrchestrationService().get_run_for_goal(
        db_session, test_project.id, replacement.id
    )
    assert replacement_run.phase == "baseline"
    assert replacement_run.budget_state == goal.budget
    assert goal.status == run.status == "cancelled"


async def test_supersede_ready_is_idempotent(db_session, test_project):
    goal, _ = await _goal(db_session, test_project, phase="ready")
    service = OrchestrationSupersessionService()

    first = await service.supersede(
        db_session, test_project.id, goal.id, new_goal_type="outcome", actor="human:test"
    )
    second = await service.supersede(
        db_session, test_project.id, goal.id, new_goal_type="outcome", actor="human:test"
    )

    replacements = list(
        await db_session.scalars(
            select(OrchestrationGoal).where(OrchestrationGoal.supersedes_goal_id == goal.id)
        )
    )
    assert second.id == first.id
    assert len(replacements) == 1


async def test_concurrent_supersedes_return_one_replacement(db_session, test_project, concurrent_sessions):
    goal, _ = await _goal(db_session, test_project, phase="ready")
    await db_session.commit()
    first_db, second_db = concurrent_sessions

    async def supersede(db):
        return await OrchestrationSupersessionService().supersede(
            db, test_project.id, goal.id, new_goal_type="outcome", actor="human:test"
        )

    first, second = await asyncio.gather(supersede(first_db), supersede(second_db))
    assert first.id == second.id

    replacements = list(
        await first_db.scalars(
            select(OrchestrationGoal).where(OrchestrationGoal.supersedes_goal_id == goal.id)
        )
    )
    assert len(replacements) == 1


async def test_supersede_route_uses_auth_autobegun_transaction(
    db_session, test_project, concurrent_sessions
):
    goal, _ = await _goal(db_session, test_project, phase="ready")
    await db_session.commit()
    route_db, _ = concurrent_sessions

    # get_current_user() performs this read when auth is enabled, so the route
    # must not call db.begin() after it has acquired the goal lock.
    assert await route_db.get(Project, test_project.id) is not None
    detail = await supersede_goal(
        test_project.id,
        goal.id,
        OrchestrationSupersedeRequest(),
        _ANON_USER,
        route_db,
    )

    assert detail.goal.supersedes_goal_id == goal.id
    assert not route_db.in_transaction()


async def test_supersede_preserves_preexisting_transaction_for_caller_rollback(
    db_session, test_project, concurrent_sessions
):
    goal, _ = await _goal(db_session, test_project, phase="ready")
    await db_session.commit()
    caller_db, verifier_db = concurrent_sessions
    caller_db.add(Task(project_id=test_project.id, title="Unrelated pending work"))
    await caller_db.flush()

    await OrchestrationSupersessionService().supersede(
        caller_db, test_project.id, goal.id, new_goal_type="outcome", actor="human:test"
    )

    assert caller_db.in_transaction()
    await caller_db.rollback()
    original = await verifier_db.get(OrchestrationGoal, goal.id)
    assert original.status == "active"
    assert await verifier_db.scalar(select(Task.id).where(Task.title == "Unrelated pending work")) is None


async def test_supersede_rejects_authorized_without_changes(db_session, test_project):
    goal, run = await _goal(db_session, test_project, phase="authorized")

    with pytest.raises(HTTPException) as exc:
        await OrchestrationSupersessionService().supersede(
            db_session, test_project.id, goal.id, new_goal_type="outcome", actor="human:test"
        )

    assert exc.value.status_code == 409
    assert exc.value.detail["conflict"] == "supersession_not_ready"
    assert goal.status == "active"
    assert run.status == "running"


async def test_supersede_rejects_completed_without_changes(db_session, test_project):
    goal, run = await _goal(db_session, test_project, phase="completed")
    goal.status = run.status = "completed"
    await db_session.flush()

    with pytest.raises(HTTPException) as exc:
        await OrchestrationSupersessionService().supersede(
            db_session, test_project.id, goal.id, new_goal_type="outcome", actor="human:test"
        )

    assert exc.value.detail["conflict"] == "supersession_not_ready"
    assert goal.status == run.status == "completed"
    assert await db_session.scalar(
        select(OrchestrationGoal.id).where(OrchestrationGoal.supersedes_goal_id == goal.id)
    ) is None


async def test_supersede_replacement_does_not_transfer_runtime_artifacts(db_session, test_project):
    goal, run = await _goal(db_session, test_project)
    db_session.add_all([
        Task(
            project_id=test_project.id,
            title="Completed original child",
            status="done",
            metadata_={"orchestration": {"run_id": str(run.id)}},
        ),
        OrchestrationAction(
            run_id=run.id,
            idempotency_key="completed-original-action",
            action_type="ask_human",
            request={},
            status="completed",
        ),
    ])
    await db_session.flush()

    replacement = await OrchestrationSupersessionService().supersede(
        db_session, test_project.id, goal.id, new_goal_type="outcome", actor="human:test"
    )
    replacement_run = await OrchestrationService().get_run_for_goal(
        db_session, test_project.id, replacement.id
    )

    assert not list(await db_session.scalars(
        select(Task.id).where(Task.metadata_["orchestration"]["run_id"].as_string() == str(replacement_run.id))
    ))
    assert not list(await db_session.scalars(
        select(OrchestrationAction.id).where(OrchestrationAction.run_id == replacement_run.id)
    ))


async def test_supersede_rejects_unfinished_child_task(db_session, test_project):
    goal, run = await _goal(db_session, test_project)
    db_session.add(Task(
        project_id=test_project.id,
        title="Active child",
        status="in_progress",
        metadata_={"orchestration": {"run_id": str(run.id)}},
    ))
    await db_session.flush()

    with pytest.raises(HTTPException) as exc:
        await OrchestrationSupersessionService().supersede(
            db_session, test_project.id, goal.id, new_goal_type="outcome", actor="human:test"
        )

    assert exc.value.detail["conflict"] == "supersession_not_ready"
    assert goal.status == "active"
    assert run.status == "running"


async def test_supersede_rejects_reserved_action(db_session, test_project):
    goal, run = await _goal(db_session, test_project)
    db_session.add(OrchestrationAction(
        run_id=run.id,
        idempotency_key="pending-action",
        action_type="ask_human",
        request={},
        status="reserved",
    ))
    await db_session.flush()

    with pytest.raises(HTTPException) as exc:
        await OrchestrationSupersessionService().supersede(
            db_session, test_project.id, goal.id, new_goal_type="outcome", actor="human:test"
        )

    assert exc.value.detail["conflict"] == "supersession_not_ready"


async def test_supersede_rejects_active_session(db_session, test_project, test_agent):
    goal, run = await _goal(db_session, test_project)
    task = Task(
        project_id=test_project.id,
        title="Finished child",
        status="done",
        metadata_={"orchestration": {"run_id": str(run.id)}},
    )
    db_session.add(task)
    await db_session.flush()
    db_session.add(Session(
        project_id=test_project.id,
        task_id=task.id,
        agent_id=test_agent.id,
        adapter_type="api",
        status="running",
    ))
    await db_session.flush()

    with pytest.raises(HTTPException) as exc:
        await OrchestrationSupersessionService().supersede(
            db_session, test_project.id, goal.id, new_goal_type="outcome", actor="human:test"
        )

    assert exc.value.detail["conflict"] == "supersession_not_ready"


async def test_supersede_rolls_back_cancellation_when_replacement_creation_fails(
    db_session, test_project, monkeypatch
):
    goal, run = await _goal(db_session, test_project)
    service = OrchestrationSupersessionService()

    async def fail_after_cancel(db, original, original_run, *, cancelled_by):
        original.status = "cancelled"
        original_run.status = "cancelled"
        await db.flush()
        raise RuntimeError("replacement failed")

    monkeypatch.setattr(service._orch, "_cancel_goal_no_commit", fail_after_cancel)
    with pytest.raises(RuntimeError, match="replacement failed"):
        await service.supersede(
            db_session, test_project.id, goal.id, new_goal_type="outcome", actor="human:test"
        )

    await db_session.refresh(goal)
    await db_session.refresh(run)
    assert goal.status == "active"
    assert run.status == "running"
