import pytest
import pytest_asyncio
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

pytestmark = pytest.mark.usefixtures("safe_goal_analysis")


@pytest.fixture(autouse=True)
def _no_live_native_dispatch(monkeypatch):
    """Keep committed semantic fixtures from launching real app workers."""
    monkeypatch.setattr("huddleroom.workers.task_runner.register_session", lambda *args, **kwargs: None)


@pytest.fixture
async def test_engine(tmp_path):
    """Isolate committed native setup, including cyclic roadmap references."""
    from sqlalchemy import event
    from huddleroom.models.base import Base
    from tests.conftest import _seed_anon_actors

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'owned-orchestration.db'}")

    @event.listens_for(engine.sync_engine, "connect")
    def foreign_keys(connection, _record):
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        await _seed_anon_actors(connection)
    yield engine
    await engine.dispose()


@pytest.fixture(autouse=True)
async def clean_test_database(test_engine):
    # Per-test database is discarded whole; no global FK cleanup is changed.
    yield


@pytest.fixture
async def db_session(test_engine):
    """This module owns committed setup for real service/worker entrypoints."""
    async with AsyncSession(test_engine, expire_on_commit=False) as db:
        yield db
        await db.rollback()


@pytest_asyncio.fixture
async def orch_goal(db_session, test_project):
    from huddleroom.models.orchestration import OrchestrationGoal

    goal = OrchestrationGoal(
        project_id=test_project.id,
        objective="Ship the widget",
        success_criteria=[{"key": "works", "description": "widget works"}],
    )
    db_session.add(goal)
    await db_session.flush()
    return goal


@pytest_asyncio.fixture
async def orch_run(db_session, orch_goal):
    from huddleroom.models.orchestration import OrchestrationRun

    run = OrchestrationRun(goal_id=orch_goal.id)
    db_session.add(run)
    await db_session.flush()
    return run


@pytest_asyncio.fixture
async def pending_decision(db_session, orch_goal, orch_run):
    from huddleroom.services.orchestration_authority_service import (
        OrchestrationAuthorityDecisionService,
    )

    return await OrchestrationAuthorityDecisionService().create_pending(
        db_session,
        orch_goal.id,
        decision_key="add-reviewer",
        title="Add an independent reviewer?",
        question="Should we add a reviewer before delegating?",
        authority="human",
        options=["add_reviewer", "continue_without"],
        recommendation="add_reviewer",
        run_id=orch_run.id,
    )


@pytest.mark.asyncio
async def test_answering_against_recommendation_creates_linked_warning(
    db_session, orch_goal, pending_decision, test_user
):
    from huddleroom.services.orchestration_authority_service import (
        OrchestrationAuthorityDecisionService,
    )
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    answered = await OrchestrationAuthorityDecisionService().answer_decision(
        db_session,
        pending_decision,
        selected_option="continue_without",
        reason="Budget does not allow a reviewer this sprint.",
        decided_by_user_id=test_user.id,
    )

    assert answered.overrides_recommendation is True
    assert answered.created_warning_id is not None
    warnings = await OrchestrationWarningService().list_warnings(db_session, orch_goal.id)
    assert len(warnings) == 1
    warning = warnings[0]
    assert warning.id == answered.created_warning_id
    assert warning.severity == "warning"
    assert warning.warning_type == "authority_decision_overrode_recommendation"
    assert warning.active is True
    assert warning.run_id == pending_decision.run_id
    assert warning.related_authority_decision_id == pending_decision.id


@pytest.mark.asyncio
async def test_tick_noops_when_hard_stop_warning_is_active(db_session, test_project):
    from huddleroom.schemas.orchestration import OrchestrationGoalCreate
    from huddleroom.services.orchestration_service import OrchestrationService
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    service = OrchestrationService()
    goal, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Ship the widget",
            success_criteria=[{"key": "works", "description": "widget works"}],
            constraints={},
            budget={},
        ),
        created_by_user_id=None,
    )
    baseline_before = await service._baseline_processes_ready_for_goal(db_session, goal.id)
    assert baseline_before is False

    await OrchestrationWarningService().create_warning(
        db_session,
        goal.id,
        warning_type="typed_action_validation_failed",
        severity="hard_stop",
        message="A typed action failed validation and orchestration must not continue.",
        run_id=run.id,
    )

    # End known-owned warning/event setup before the real tick.
    await db_session.flush()
    await db_session.commit()
    result = await service.tick(db_session, run.id)

    await db_session.refresh(goal)
    await db_session.refresh(run)
    assert result["baseline_process"] is None
    assert result["run_completed"] is False
    assert goal.status == "active"
    assert run.status == "running"

    warning = (
        await OrchestrationWarningService().list_warnings(db_session, goal.id, active_only=True)
    )[0]
    await OrchestrationWarningService().resolve_warning(
        db_session, warning, resolved_by="human:tester", reason="False alarm, action was retried."
    )
    # End known-owned warning/event setup before the real tick.
    await db_session.flush()
    await db_session.commit()
    result_after = await service.tick(db_session, run.id)
    assert result_after["baseline_process"] is None
    await db_session.refresh(run)
    assert run.baseline_authorized is False  # Resolving a warning grants no Continue authority.
    await service.authorize_baseline(db_session, test_project.id, goal.id, actor="human:tester")
    await db_session.commit()
    continued = await service.tick(db_session, run.id)
    assert continued["baseline_process"] is not None


@pytest.mark.asyncio
async def test_completion_manifest_allows_completion_once_ordinary_warning_acknowledged(
    db_session, test_project
):
    from huddleroom.schemas.orchestration import OrchestrationGoalCreate
    from huddleroom.services.orchestration_service import OrchestrationService
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService
    from tests.conftest import complete_baseline_processes

    service = OrchestrationService()
    goal, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Ship the widget",
            success_criteria=[{"key": "works", "description": "widget works"}],
            constraints={},
            budget={},
        ),
        created_by_user_id=None,
    )
    run.baseline_authorized = True
    await complete_baseline_processes(db_session, goal, run)
    warning_service = OrchestrationWarningService()
    warning = await warning_service.create_warning(
        db_session,
        goal.id,
        warning_type="goal_definition_skipped",
        severity="warning",
        message="Goal clarification was skipped by human override.",
    )

    with pytest.raises(HTTPException, match="must be acknowledged or resolved"):
        await service._closeout_preconditions_manifest(db_session, goal, run)

    await warning_service.acknowledge_warning(db_session, warning, acknowledged_by="human:tester")
    with pytest.raises(HTTPException, match="All orchestration gates must be accepted"):
        await service._closeout_preconditions_manifest(db_session, goal, run)


@pytest.mark.asyncio
async def test_completion_manifest_allows_completion_once_blocker_resolved(
    db_session, test_project
):
    from huddleroom.schemas.orchestration import OrchestrationGoalCreate
    from huddleroom.services.orchestration_service import OrchestrationService
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService
    from tests.conftest import complete_baseline_processes

    service = OrchestrationService()
    goal, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Ship the widget",
            success_criteria=[{"key": "works", "description": "widget works"}],
            constraints={},
            budget={},
        ),
        created_by_user_id=None,
    )
    run.baseline_authorized = True
    await complete_baseline_processes(db_session, goal, run)
    warning_service = OrchestrationWarningService()
    warning = await warning_service.create_warning(
        db_session,
        goal.id,
        warning_type="no_independent_verifier",
        severity="blocker",
        message="No safe independent verifier was found.",
        run_id=run.id,
    )

    await db_session.refresh(run)
    assert f"warning:{warning.id}" in [item["kind"] for item in run.active_blockers]

    # End known-owned warning/event setup before the real tick.
    await db_session.flush()
    await db_session.commit()
    result = await service.tick(db_session, run.id)
    assert result["run_completed"] is False

    with pytest.raises(HTTPException, match="must be resolved"):
        await service._closeout_preconditions_manifest(db_session, goal, run)

    await warning_service.resolve_warning(
        db_session, warning, resolved_by="human:tester", reason="Verifier was added."
    )

    await db_session.refresh(run)
    assert run.active_blockers == []
    with pytest.raises(HTTPException, match="All orchestration gates must be accepted"):
        await service._closeout_preconditions_manifest(db_session, goal, run)


@pytest.mark.asyncio
async def test_warnings_endpoints_list_acknowledge_resolve(
    client, db_session, orch_goal, orch_run, test_project, auth_headers
):
    from huddleroom.services.orchestration_authority_service import (
        OrchestrationAuthorityDecisionService,
    )
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    warning = await OrchestrationWarningService().create_warning(
        db_session,
        orch_goal.id,
        warning_type="no_manager_selected",
        severity="warning",
        message="No manager was selected.",
    )
    await db_session.flush()

    list_response = await client.get(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{orch_goal.id}/warnings",
        headers=auth_headers,
    )
    assert list_response.status_code == 200
    body = list_response.json()
    assert len(body) == 1
    assert body[0]["id"] == str(warning.id)
    assert body[0]["blocks_completion"] is True

    ack_response = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/warnings/{warning.id}/acknowledge",
        json={"reason": "Acceptable risk for this sprint."},
        headers=auth_headers,
    )
    assert ack_response.status_code == 200
    assert ack_response.json()["acknowledged_at"] is not None
    assert ack_response.json()["blocks_completion"] is False

    decisions = await OrchestrationAuthorityDecisionService().list_decisions(
        db_session, orch_goal.id
    )
    ack_decision = next(
        decision
        for decision in decisions
        if decision.decision_key == f"acknowledge-warning-{warning.id}"
    )
    assert ack_decision.status == "answered"
    assert ack_decision.selected_option == "acknowledge"
    assert ack_decision.reason == "Acceptable risk for this sprint."

    resolve_response = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/warnings/{warning.id}/resolve",
        json={"reason": "Manager was assigned."},
        headers=auth_headers,
    )
    assert resolve_response.status_code == 200
    assert resolve_response.json()["active"] is False

    filtered_response = await client.get(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{orch_goal.id}/warnings"
        "?active_only=true",
        headers=auth_headers,
    )
    assert filtered_response.json() == []
