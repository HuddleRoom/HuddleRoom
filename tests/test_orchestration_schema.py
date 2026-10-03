import os
import subprocess
import sys
import uuid

import pytest
import sqlalchemy as sa
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError


def test_migration_016_creates_orchestration_schema_contract(tmp_path):
    db_path = tmp_path / "migrated.db"
    db_url = f"sqlite+aiosqlite:///{db_path}"
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "RALLY_DATABASE_URL": db_url},
    )
    assert result.returncode == 0, result.stderr

    engine = create_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        inspector = sa.inspect(conn)
        assert {"orchestration_goals", "orchestration_runs"} <= set(inspector.get_table_names())

        goal_columns = {column["name"] for column in inspector.get_columns("orchestration_goals")}
        assert {
            "id",
            "project_id",
            "objective",
            "success_criteria",
            "constraints",
            "budget",
            "status",
            "created_by_user_id",
            "created_at",
            "updated_at",
        } <= goal_columns

        run_columns = {column["name"] for column in inspector.get_columns("orchestration_runs")}
        assert {
            "id",
            "goal_id",
            "status",
            "event_cursor",
            "plan_state",
            "active_blockers",
            "budget_state",
            "retry_state",
            "started_at",
            "completed_at",
            "created_at",
            "updated_at",
        } <= run_columns

        goal_checks = {check["name"] for check in inspector.get_check_constraints("orchestration_goals")}
        run_checks = {check["name"] for check in inspector.get_check_constraints("orchestration_runs")}
        assert "ck_orchestration_goals_status" in goal_checks
        assert "ck_orchestration_runs_status" in run_checks

        goal_indexes = {index["name"] for index in inspector.get_indexes("orchestration_goals")}
        run_indexes = {index["name"] for index in inspector.get_indexes("orchestration_runs")}
        assert {"idx_orch_goals_project_status", "idx_orch_goals_project_created"} <= goal_indexes
        assert {"idx_orch_runs_goal_status", "uq_orch_runs_one_active_per_goal"} <= run_indexes
        assert "idx_orch_runs_status_updated" not in run_indexes

        active_index_sql = conn.exec_driver_sql(
            "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = 'uq_orch_runs_one_active_per_goal'"
        ).scalar_one()
        assert "WHERE status IN ('running', 'blocked', 'paused')" in active_index_sql


def test_orchestration_model_timestamps_match_migration_timezone_contract():
    from huddleroom.models.base import TimestampMixin
    from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun

    assert OrchestrationGoal.__table__.c.created_at.type.timezone is True
    assert OrchestrationGoal.__table__.c.updated_at.type.timezone is True
    assert OrchestrationRun.__table__.c.started_at.type.timezone is True
    assert OrchestrationRun.__table__.c.completed_at.type.timezone is True
    assert OrchestrationRun.__table__.c.created_at.type.timezone is True
    assert OrchestrationRun.__table__.c.updated_at.type.timezone is True
    assert TimestampMixin not in OrchestrationGoal.__mro__
    assert TimestampMixin not in OrchestrationRun.__mro__


@pytest.mark.asyncio
async def test_goal_and_run_status_constraints(db_session, test_project):
    from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun

    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            db_session.add(
                OrchestrationGoal(
                    project_id=test_project.id,
                    objective="Invalid goal status",
                    success_criteria=[{"key": "done", "description": "Done"}],
                    constraints={},
                    budget={},
                    status="not-real",
                )
            )
            await db_session.flush()

    valid_goal = OrchestrationGoal(
        project_id=test_project.id,
        objective="Valid goal",
        success_criteria=[{"key": "done", "description": "Done"}],
        constraints={},
        budget={},
    )
    db_session.add(valid_goal)
    await db_session.flush()

    first_run = OrchestrationRun(
        goal_id=valid_goal.id,
        event_cursor=None,
        plan_state={},
        active_blockers=[],
        budget_state={},
        retry_state={},
    )
    db_session.add(first_run)
    await db_session.flush()

    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            db_session.add(
                OrchestrationRun(
                    goal_id=valid_goal.id,
                    event_cursor=None,
                    plan_state={},
                    active_blockers=[],
                    budget_state={},
                    retry_state={},
                )
            )
            await db_session.flush()

    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            db_session.add(
                OrchestrationRun(
                    goal_id=valid_goal.id,
                    status="not-real",
                    event_cursor=None,
                    plan_state={},
                    active_blockers=[],
                    budget_state={},
                    retry_state={},
                )
            )
            await db_session.flush()


def test_goal_create_schema_accepts_empty_success_criteria():
    """An unclear goal is accepted at
    intake and clarified through the goal-definition process (which raises a
    goal_definition:success_criteria_missing decision), not rejected with a
    422. Supersedes the pre-Phase-5 min_length=1 behavior this test used to
    pin."""
    from huddleroom.schemas.orchestration import OrchestrationGoalCreate

    data = OrchestrationGoalCreate(
        objective="Ship the orchestration foundation",
        success_criteria=[],
    )
    assert data.success_criteria == []


@pytest.mark.asyncio
async def test_service_creates_goal_with_one_run(db_session, test_project):
    from huddleroom.models.project import Project
    from huddleroom.schemas.orchestration import OrchestrationGoalCreate
    from huddleroom.services.orchestration_service import OrchestrationService

    service = OrchestrationService()
    data = OrchestrationGoalCreate(
        objective="Ship an orchestration goal",
        success_criteria=[
            {
                "key": "schema_exists",
                "description": "Goals and runs are stored durably.",
            }
        ],
        constraints={"limits": {"scope": "schema-only"}},
        budget={"caps": {"max_tokens": 1000}},
    )

    goal, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=data,
        created_by_user_id=None,
    )

    assert goal.id is not None
    assert goal.project_id == test_project.id
    assert goal.objective == "Ship an orchestration goal"
    assert goal.success_criteria[0]["key"] == "schema_exists"
    assert goal.constraints == {"limits": {"scope": "schema-only"}}
    assert goal.budget == {"caps": {"max_tokens": 1000}}
    assert goal.status == "active"

    assert run.id is not None
    assert run.goal_id == goal.id
    assert run.status == "running"
    assert run.event_cursor is None
    assert run.plan_state == {}
    assert run.active_blockers == []
    assert run.budget_state == {"caps": {"max_tokens": 1000}}
    assert run.retry_state == {}
    assert run.started_at is not None
    assert run.completed_at is None

    data.budget["caps"]["max_tokens"] = 2000
    data.success_criteria[0]["key"] = "mutated"
    data.constraints["limits"]["scope"] = "changed"
    data.success_criteria.append({"key": "later", "description": "Added later."})
    assert goal.budget == {"caps": {"max_tokens": 1000}}
    assert run.budget_state == {"caps": {"max_tokens": 1000}}
    assert goal.success_criteria == [
        {
            "key": "schema_exists",
            "description": "Goals and runs are stored durably.",
        }
    ]
    assert goal.constraints == {"limits": {"scope": "schema-only"}}

    loaded_goal = await service.get_goal(db_session, test_project.id, goal.id)
    assert loaded_goal is not None
    assert loaded_goal.id == goal.id

    active_run = await service.get_active_run_for_goal(db_session, test_project.id, goal.id)
    assert active_run is not None
    assert active_run.id == run.id

    other_project = Project(name="Other Project", description="", config={})
    db_session.add(other_project)
    await db_session.flush()

    wrong_scope_run = await service.get_active_run_for_goal(db_session, other_project.id, goal.id)
    assert wrong_scope_run is None


@pytest.mark.asyncio
async def test_service_lists_goals_by_project_and_status(db_session, test_project):
    from huddleroom.schemas.orchestration import OrchestrationGoalCreate
    from huddleroom.services.orchestration_service import OrchestrationService

    service = OrchestrationService()
    first, _ = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="First goal",
            success_criteria=[{"key": "first", "description": "First criterion"}],
        ),
        created_by_user_id=None,
    )
    second, _ = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Second goal",
            success_criteria=[{"key": "second", "description": "Second criterion"}],
        ),
        created_by_user_id=None,
    )

    first_page, next_cursor = await service.list_goals(db_session, test_project.id, limit=1)
    assert len(first_page) == 1
    assert next_cursor is not None

    second_page, final_cursor = await service.list_goals(db_session, test_project.id, cursor=next_cursor, limit=10)
    assert final_cursor is None

    goal_ids = {goal.id for goal in [*first_page, *second_page]}
    assert first.id in goal_ids
    assert second.id in goal_ids

    active_goals, _ = await service.list_goals(db_session, test_project.id, status="active")
    assert {goal.status for goal in active_goals} == {"active"}

    missing = await service.get_goal(db_session, test_project.id, uuid.uuid4())
    assert missing is None


@pytest.mark.parametrize(
    ("schema_class", "kwargs"),
    [
        ("OrchestrationGoalWeightOverrideRequest", {"weight": "substantial", "reason": "   "}),
        ("OrchestrationProcessStartRequest", {"reason": "   "}),
        ("OrchestrationProcessSkipRequest", {"reason": "   "}),
    ],
)
def test_whitespace_reason_is_rejected(schema_class, kwargs):
    from huddleroom.schemas.orchestration import (
        OrchestrationGoalWeightOverrideRequest,
        OrchestrationProcessStartRequest,
        OrchestrationProcessSkipRequest,
    )

    schema_map = {
        "OrchestrationGoalWeightOverrideRequest": OrchestrationGoalWeightOverrideRequest,
        "OrchestrationProcessStartRequest": OrchestrationProcessStartRequest,
        "OrchestrationProcessSkipRequest": OrchestrationProcessSkipRequest,
    }
    with pytest.raises(ValidationError):
        schema_map[schema_class](**kwargs)


@pytest.mark.asyncio
async def test_goal_no_manager_constraint_forbids_manager_ids_with_no_manager(db_session, test_project, test_agent):
    """no_manager constraint forbids authority_model='no_manager' when manager IDs are set."""
    from huddleroom.models.orchestration import OrchestrationGoal

    # Test with manager_agent_id set
    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            db_session.add(
                OrchestrationGoal(
                    project_id=test_project.id,
                    objective="Goal with no_manager but agent set",
                    success_criteria=[],
                    constraints={},
                    budget={},
                    authority_model="no_manager",
                    manager_agent_id=test_agent.id,
                )
            )
            await db_session.flush()


@pytest.mark.asyncio
async def test_goal_no_manager_constraint_allows_valid_states(db_session, test_project, test_agent):
    """no_manager constraint allows valid states: no_manager with no IDs, and manager models with NULL IDs (transient state)."""
    from huddleroom.models.orchestration import OrchestrationGoal

    # Test: authority_model='no_manager' with both IDs NULL (valid)
    valid_no_manager = OrchestrationGoal(
        project_id=test_project.id,
        objective="Valid no_manager goal",
        success_criteria=[],
        constraints={},
        budget={},
        authority_model="no_manager",
        manager_agent_id=None,
        manager_user_id=None,
    )
    db_session.add(valid_no_manager)
    await db_session.flush()
    assert valid_no_manager.id is not None

    # Test: authority_model='agent_manager' with NULL manager_agent_id (transient deleted-manager state)
    transient_agent_manager = OrchestrationGoal(
        project_id=test_project.id,
        objective="Transient agent_manager with NULL manager_agent_id",
        success_criteria=[],
        constraints={},
        budget={},
        authority_model="agent_manager",
        manager_agent_id=None,
        manager_user_id=None,
    )
    db_session.add(transient_agent_manager)
    await db_session.flush()
    assert transient_agent_manager.id is not None

    # Test: authority_model='human_manager' with NULL manager_user_id (transient deleted-manager state)
    transient_human_manager = OrchestrationGoal(
        project_id=test_project.id,
        objective="Transient human_manager with NULL manager_user_id",
        success_criteria=[],
        constraints={},
        budget={},
        authority_model="human_manager",
        manager_agent_id=None,
        manager_user_id=None,
    )
    db_session.add(transient_human_manager)
    await db_session.flush()
    assert transient_human_manager.id is not None

    # Test: authority_model=NULL (before manager selection completes)
    null_authority = OrchestrationGoal(
        project_id=test_project.id,
        objective="Goal before manager selection",
        success_criteria=[],
        constraints={},
        budget={},
        authority_model=None,
        manager_agent_id=None,
        manager_user_id=None,
    )
    db_session.add(null_authority)
    await db_session.flush()
    assert null_authority.id is not None
