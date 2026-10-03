import os
import sqlite3
import subprocess
import sys
import uuid

import pytest
import sqlalchemy as sa

from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun
from huddleroom.services.orchestration_debug_service import OrchestrationDebugService
from huddleroom.services.orchestration_goal_definition import GoalDefinitionProcess
from huddleroom.services.orchestration_process_service import OrchestrationProcessService
from huddleroom.services.orchestration_service import OrchestrationService


def _alembic(env: dict[str, str], *command: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "alembic", *command],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr


def test_migration_046_backfills_baseline_authorized_false_for_baseline_runs(tmp_path):
    db_path = tmp_path / "migration-046.db"
    env = {**os.environ, "RALLY_DATABASE_URL": f"sqlite+aiosqlite:///{db_path}"}
    _alembic(env, "upgrade", "045")

    now = "2026-09-23T00:00:00+00:00"
    goal_ids = {"baseline": str(uuid.uuid4()), "ready": str(uuid.uuid4())}
    with sqlite3.connect(db_path) as connection:
        for phase, goal_id in goal_ids.items():
            connection.execute(
                """
                INSERT INTO orchestration_runs (
                    id, goal_id, status, event_cursor, plan_state, active_blockers, budget_state,
                    retry_state, phase, started_at, completed_at, created_at, updated_at, supervision_state
                ) VALUES (?, ?, 'running', NULL, '{}', '[]', '{}', '{}', ?, ?, NULL, ?, ?, '{}')
                """,
                (str(uuid.uuid4()), goal_id, phase, now, now, now),
            )
        connection.commit()

    _alembic(env, "upgrade", "046")

    with sqlite3.connect(db_path) as connection:
        inspector = sa.inspect(sa.create_engine(f"sqlite:///{db_path}"))
        columns = {c["name"]: (c["default"], c["nullable"]) for c in inspector.get_columns("orchestration_runs")}
        assert columns["baseline_authorized"] == ("true", False)

        rows = dict(connection.execute(
            "SELECT phase, baseline_authorized FROM orchestration_runs"
        ).fetchall())
        assert rows["baseline"] == 0
        assert rows["ready"] == 1


def test_migration_046_round_trip_restores_045_schema(tmp_path):
    db_path = tmp_path / "migration-046-round-trip.db"
    env = {**os.environ, "RALLY_DATABASE_URL": f"sqlite+aiosqlite:///{db_path}"}
    _alembic(env, "upgrade", "046")
    _alembic(env, "downgrade", "045")

    inspector = sa.inspect(sa.create_engine(f"sqlite:///{db_path}"))
    assert "baseline_authorized" not in {
        column["name"] for column in inspector.get_columns("orchestration_runs")
    }


@pytest.mark.asyncio
async def test_orm_default_is_baseline_authorized_true(db_session, test_project):
    """44+ test files construct OrchestrationRun directly and rely on this default."""
    goal = OrchestrationGoal(
        project_id=test_project.id, objective="Ship X", original_request="Ship X",
        success_criteria=[], constraints={}, budget={},
    )
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id, budget_state={})
    db_session.add(run)
    await db_session.flush()

    assert run.baseline_authorized is True


@pytest.mark.asyncio
async def test_create_goal_starts_gated_and_authorize_unlocks_tick(
    db_session, test_project, safe_goal_analysis
):
    service = OrchestrationService()
    from huddleroom.schemas.orchestration import OrchestrationGoalCreate

    goal, run = await service.create_goal(
        db_session,
        test_project.id,
        OrchestrationGoalCreate(objective="Ship X", success_criteria=[], constraints={}, budget={}),
        created_by_user_id=None,
    )
    assert run.baseline_authorized is False

    result = await GoalDefinitionProcess().advance(db_session, goal, run)
    assert result["status"] == "not_started"
    current = await OrchestrationProcessService().get_current(db_session, goal.id, "goal_definition")
    assert current is None

    goal, run = await service.authorize_baseline(db_session, test_project.id, goal.id, actor="human:test")
    assert run.baseline_authorized is True

    # Idempotent: calling again is a no-op, not an error.
    goal, run = await service.authorize_baseline(db_session, test_project.id, goal.id, actor="human:test")
    assert run.baseline_authorized is True


@pytest.mark.asyncio
async def test_authorize_baseline_404s_past_baseline_phase(db_session, test_project):
    goal = OrchestrationGoal(
        project_id=test_project.id, objective="Ship X", original_request="Ship X",
        success_criteria=[], constraints={}, budget={},
    )
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id, budget_state={}, phase="ready", baseline_authorized=True)
    db_session.add(run)
    await db_session.flush()

    with pytest.raises(Exception) as exc_info:
        await OrchestrationService().authorize_baseline(db_session, test_project.id, goal.id, actor="human:test")
    assert getattr(exc_info.value, "status_code", None) == 409


@pytest.mark.asyncio
async def test_pause_mid_baseline_clears_authorization(db_session, test_project):
    service = OrchestrationService()
    from huddleroom.schemas.orchestration import OrchestrationGoalCreate

    goal, run = await service.create_goal(
        db_session,
        test_project.id,
        OrchestrationGoalCreate(objective="Ship X", success_criteria=[], constraints={}, budget={}),
        created_by_user_id=None,
    )
    await service.authorize_baseline(db_session, test_project.id, goal.id, actor="human:test")
    await db_session.refresh(run)
    assert run.baseline_authorized is True

    await service.pause_goal(db_session, test_project.id, goal.id)
    await db_session.refresh(run)
    assert run.baseline_authorized is False


@pytest.mark.asyncio
async def test_debug_dispatcher_step_still_works_when_unauthorized(
    db_session, test_project, safe_goal_analysis
):
    service = OrchestrationService()
    from huddleroom.schemas.orchestration import OrchestrationGoalCreate

    goal, run = await service.create_goal(
        db_session,
        test_project.id,
        OrchestrationGoalCreate(objective="Ship X", success_criteria=[], constraints={}, budget={}),
        created_by_user_id=None,
    )
    assert run.baseline_authorized is False

    result = await OrchestrationDebugService().advance_process(db_session, goal, run, "goal_definition")
    assert result["process_type"] == "goal_definition"
    current = await OrchestrationProcessService().get_current(db_session, goal.id, "goal_definition")
    assert current is not None
