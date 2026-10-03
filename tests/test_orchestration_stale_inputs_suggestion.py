import pytest
import pytest_asyncio

pytestmark = pytest.mark.usefixtures("safe_goal_analysis")


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
async def process_run(db_session, orch_goal, orch_run):
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    return await OrchestrationProcessService().start_process(
        db_session,
        orch_goal.id,
        process_type="agent_definition_review",
        trigger_reason="tick: no agent definition review on record",
        run_id=orch_run.id,
        input_snapshot={"fingerprint": "abc"},
        process_version=3,
    )


@pytest.mark.asyncio
async def test_suggest_stale_inputs_dedupes_including_resolved(
    db_session, orch_goal, orch_run, process_run
):
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    service = OrchestrationWarningService()
    first = await service.suggest_stale_inputs(
        db_session,
        orch_goal.id,
        process_type="agent_definition_review",
        process_run_id=process_run.id,
        step_label="agent definition review",
        run_id=orch_run.id,
    )
    warnings = await service.list_warnings(db_session, orch_goal.id)
    assert len(warnings) == 1
    assert first.warning_type == "agent_definition_review_stale_inputs"
    assert first.severity == "recommendation"

    # Second call with the same process_run_id: no-op, even before resolution.
    again = await service.suggest_stale_inputs(
        db_session,
        orch_goal.id,
        process_type="agent_definition_review",
        process_run_id=process_run.id,
        step_label="agent definition review",
        run_id=orch_run.id,
    )
    assert again.id == first.id
    assert len(await service.list_warnings(db_session, orch_goal.id)) == 1

    # Dismiss it (resolved), then call again: must NOT resurface as a new row.
    await service.resolve_warning(
        db_session, first, resolved_by="human:1", reason="dismissed by human"
    )
    dismissed_again = await service.suggest_stale_inputs(
        db_session,
        orch_goal.id,
        process_type="agent_definition_review",
        process_run_id=process_run.id,
        step_label="agent definition review",
        run_id=orch_run.id,
    )
    assert dismissed_again.id == first.id
    assert dismissed_again.active is False
    all_warnings = await service.list_warnings(db_session, orch_goal.id)
    assert len(all_warnings) == 1


@pytest.mark.asyncio
async def test_suggest_stale_inputs_new_process_run_gets_own_warning(
    db_session, orch_goal, orch_run, process_run
):
    from huddleroom.models.orchestration_process import OrchestrationProcessRun
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    service = OrchestrationWarningService()
    await service.suggest_stale_inputs(
        db_session,
        orch_goal.id,
        process_type="agent_definition_review",
        process_run_id=process_run.id,
        step_label="agent definition review",
        run_id=orch_run.id,
    )

    process_run.status = "completed"
    # Self-referencing superseded_by_id satisfies the partial unique index
    # on (goal_id, process_type) WHERE superseded_by_id IS NULL, freeing up
    # the slot for the second row below -- same pattern the real supersede
    # flow uses before pointing it at the actual successor.
    process_run.superseded_by_id = process_run.id
    await db_session.flush()

    # A real rerun produces a brand-new process run row (independent id).
    second_run = OrchestrationProcessRun(
        goal_id=orch_goal.id,
        run_id=orch_run.id,
        process_type="agent_definition_review",
        process_version=3,
        status="completed",
        trigger_reason="approved rerun",
        input_snapshot={"fingerprint": "def"},
    )
    db_session.add(second_run)
    await db_session.flush()
    await service.suggest_stale_inputs(
        db_session,
        orch_goal.id,
        process_type="agent_definition_review",
        process_run_id=second_run.id,
        step_label="agent definition review",
        run_id=orch_run.id,
    )
    warnings = await service.list_warnings(db_session, orch_goal.id)
    assert len(warnings) == 2
    assert {w.source_process_run_id for w in warnings} == {process_run.id, second_run.id}
