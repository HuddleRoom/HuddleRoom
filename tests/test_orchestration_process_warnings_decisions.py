import uuid

import pytest
import pytest_asyncio
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError

from huddleroom.models.base import _utcnow


async def _table_names(engine):
    async with engine.connect() as conn:
        return await conn.run_sync(lambda sync_conn: inspect(sync_conn).get_table_names())


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


@pytest.mark.asyncio
async def test_process_run_insert_defaults(db_session, orch_goal, orch_run):
    from huddleroom.models.orchestration_process import OrchestrationProcessRun

    row = OrchestrationProcessRun(
        goal_id=orch_goal.id,
        run_id=orch_run.id,
        process_type="goal_definition",
        trigger_reason="new goal created",
    )
    db_session.add(row)
    await db_session.flush()

    assert row.id is not None
    assert row.status == "running"
    assert row.process_version == 1
    assert row.input_snapshot == {}
    assert row.outputs == {}
    assert row.skipped_by is None
    assert row.override_reason is None
    assert row.superseded_by_id is None
    assert row.started_at is not None
    assert row.completed_at is None


@pytest.mark.asyncio
async def test_process_run_bad_status_rejected(db_session, orch_goal):
    from huddleroom.models.orchestration_process import OrchestrationProcessRun

    row = OrchestrationProcessRun(
        goal_id=orch_goal.id,
        process_type="goal_definition",
        trigger_reason="x",
        status="bogus",
    )
    db_session.add(row)
    with pytest.raises(IntegrityError):
        await db_session.flush()


@pytest.mark.asyncio
async def test_warning_insert_defaults_and_links(db_session, orch_goal, orch_run):
    from huddleroom.models.orchestration_process import (
        OrchestrationProcessRun,
        OrchestrationWarning,
    )

    process_run = OrchestrationProcessRun(
        goal_id=orch_goal.id,
        process_type="team_hierarchy",
        trigger_reason="manager selected",
    )
    db_session.add(process_run)
    await db_session.flush()

    warning = OrchestrationWarning(
        goal_id=orch_goal.id,
        run_id=orch_run.id,
        warning_type="no_independent_verifier",
        severity="warning",
        message="No safe independent verifier was found.",
        source_process_run_id=process_run.id,
    )
    db_session.add(warning)
    await db_session.flush()

    assert warning.id is not None
    assert warning.active is True or warning.active == 1  # SQLite may return int
    assert warning.goal_id == orch_goal.id
    assert warning.run_id == orch_run.id
    assert warning.source_process_run_id == process_run.id
    assert warning.source_agent_review_id is None
    assert warning.acknowledged_by is None
    assert warning.resolved_by is None
    assert warning.resolved_reason is None


@pytest.mark.asyncio
async def test_warning_bad_severity_rejected(db_session, orch_goal):
    from huddleroom.models.orchestration_process import OrchestrationWarning

    warning = OrchestrationWarning(
        goal_id=orch_goal.id,
        warning_type="x",
        severity="catastrophic",
        message="nope",
    )
    db_session.add(warning)
    with pytest.raises(IntegrityError):
        await db_session.flush()




@pytest.mark.asyncio
async def test_authority_decision_insert_defaults(db_session, orch_goal, orch_run):
    from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision

    decision = OrchestrationAuthorityDecision(
        goal_id=orch_goal.id,
        run_id=orch_run.id,
        decision_key="goal_definition:success_criteria",
        title="Confirm success criteria",
        authority="human",
        question="Are these success criteria complete?",
    )
    db_session.add(decision)
    await db_session.flush()

    assert decision.id is not None
    assert decision.status == "pending"
    assert decision.options == []
    assert decision.overrides_recommendation is False or decision.overrides_recommendation == 0
    assert decision.selected_option is None
    assert decision.decided_at is None
    assert decision.asked_at is not None
    assert decision.authority_agent_id is None
    assert decision.consequences is None


@pytest.mark.asyncio
async def test_authority_decision_key_unique_per_goal_while_pending(db_session, orch_goal):
    from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision

    def make():
        return OrchestrationAuthorityDecision(
            goal_id=orch_goal.id,
            decision_key="manager_selection:candidate",
            title="Approve manager",
            authority="human",
            question="Approve suggested manager?",
        )

    db_session.add(make())
    await db_session.flush()
    db_session.add(make())
    with pytest.raises(IntegrityError):
        await db_session.flush()


@pytest.mark.asyncio
async def test_authority_decision_key_reusable_after_terminal(db_session, orch_goal):
    # Partial unique index only covers status='pending' (spec 6.3: a rerun
    # must be able to raise a fresh decision under the same logical key once
    # the prior one is answered/cancelled).
    from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision

    def make():
        return OrchestrationAuthorityDecision(
            goal_id=orch_goal.id,
            decision_key="manager_selection:candidate2",
            title="Approve manager",
            authority="human",
            question="Approve suggested manager?",
            status="cancelled",
        )

    db_session.add(make())
    await db_session.flush()
    db_session.add(make())
    await db_session.flush()  # no IntegrityError: first row is terminal


@pytest.mark.asyncio
async def test_authority_decision_bad_authority_rejected(db_session, orch_goal):
    from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision

    decision = OrchestrationAuthorityDecision(
        goal_id=orch_goal.id,
        decision_key="x",
        title="x",
        authority="committee",
        question="?",
    )
    db_session.add(decision)
    with pytest.raises(IntegrityError):
        await db_session.flush()


@pytest.mark.asyncio
async def test_goal_weight_defaults(db_session, orch_goal):
    assert orch_goal.weight == "standard"
    assert orch_goal.weight_overridden_by is None


@pytest.mark.asyncio
async def test_goal_bad_weight_rejected(db_session, test_project):
    # CHECK comes from the model via create_all on the test engine
    # (Postgres-migrated DBs get it in migration 022; migrated SQLite DBs
    # rely on Phase 5 service validation).
    from huddleroom.models.orchestration import OrchestrationGoal

    goal = OrchestrationGoal(
        project_id=test_project.id,
        objective="x",
        success_criteria=[],
        weight="gigantic",
    )
    db_session.add(goal)
    with pytest.raises(IntegrityError):
        await db_session.flush()


@pytest.mark.asyncio
async def test_process_start_complete(db_session, orch_goal, orch_run):
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    svc = OrchestrationProcessService()
    row = await svc.start_process(
        db_session, orch_goal.id,
        process_type="goal_definition",
        trigger_reason="new goal created",
        run_id=orch_run.id,
    )
    assert row.status == "running"
    assert row.trigger_reason == "new goal created"

    done = await svc.complete_process(db_session, row, outputs={"open_questions": 2})
    assert done.status == "completed"
    assert done.outputs == {"open_questions": 2}
    assert done.completed_at is not None


@pytest.mark.asyncio
async def test_process_complete_requires_running(db_session, orch_goal):
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    svc = OrchestrationProcessService()
    row = await svc.start_process(
        db_session, orch_goal.id, process_type="goal_definition", trigger_reason="x",
    )
    await svc.complete_process(db_session, row)
    with pytest.raises(ValueError):
        await svc.complete_process(db_session, row)


@pytest.mark.asyncio
async def test_process_start_is_idempotent_while_running(db_session, orch_goal):
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    svc = OrchestrationProcessService()
    first = await svc.start_process(
        db_session, orch_goal.id, process_type="goal_definition", trigger_reason="new goal created",
    )
    again = await svc.start_process(
        db_session, orch_goal.id, process_type="goal_definition", trigger_reason="new goal created",
    )
    assert again.id == first.id
    assert first.superseded_by_id is None
    runs = await svc.list_process_runs(db_session, orch_goal.id, process_type="goal_definition")
    assert len(runs) == 1


@pytest.mark.asyncio
async def test_process_start_is_idempotent_while_waiting_decision(db_session, orch_goal):
    # A current run in waiting_decision status (parked process retried)
    # should return the parked row, not spawn a duplicate (Finding 1).
    from sqlalchemy import update

    from huddleroom.models.orchestration_process import OrchestrationProcessRun
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    svc = OrchestrationProcessService()
    first = await svc.start_process(
        db_session, orch_goal.id, process_type="goal_definition", trigger_reason="new goal created",
    )
    # Simulate a parked process by manually setting status to waiting_decision
    await db_session.execute(
        update(OrchestrationProcessRun)
        .where(OrchestrationProcessRun.id == first.id)
        .values(status="waiting_decision")
    )
    await db_session.flush()
    # Retry start_process with the same process_type
    again = await svc.start_process(
        db_session, orch_goal.id, process_type="goal_definition", trigger_reason="new goal created",
    )
    # Must return the SAME row (same id), not create a new one
    assert again.id == first.id
    assert again.status == "waiting_decision"  # Status unchanged
    assert first.superseded_by_id is None
    runs = await svc.list_process_runs(db_session, orch_goal.id, process_type="goal_definition")
    assert len(runs) == 1


async def _goal_warnings(db_session, goal_id):
    from sqlalchemy import select

    from huddleroom.models.orchestration_process import OrchestrationWarning

    result = await db_session.execute(
        select(OrchestrationWarning).where(OrchestrationWarning.goal_id == goal_id)
    )
    return list(result.scalars().all())


@pytest.mark.asyncio
async def test_process_skip_records_who_and_reason(db_session, orch_goal):
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    svc = OrchestrationProcessService()
    row = await svc.skip_process(
        db_session, orch_goal.id,
        process_type="team_hierarchy",
        skipped_by="human:1234",
        reason="small team, hierarchy obvious",
    )
    assert row.status == "skipped"
    assert row.skipped_by == "human:1234"
    assert row.override_reason == "small team, hierarchy obvious"
    assert row.completed_at is not None
    with pytest.raises(ValueError):
        await svc.complete_process(db_session, row)

    # skip_process must create the process's skip warning inline (spec 6.5) —
    # this test only touches the OrchestrationWarning model (Task 1), not the
    # Task 4 warning service, to keep task ordering independent.
    warnings = await _goal_warnings(db_session, orch_goal.id)
    assert {warning.warning_type for warning in warnings} == {
        "team_hierarchy_skipped",
        "team_hierarchy_not_reviewed",
    }
    assert all(warning.source_process_run_id == row.id for warning in warnings)
    assert all(bool(warning.active) is True for warning in warnings)


@pytest.mark.asyncio
async def test_process_skip_is_idempotent(db_session, orch_goal):
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    svc = OrchestrationProcessService()
    kwargs = dict(
        process_type="team_hierarchy", skipped_by="human:1234", reason="small team",
    )
    first = await svc.skip_process(db_session, orch_goal.id, **kwargs)
    again = await svc.skip_process(db_session, orch_goal.id, **kwargs)
    assert again.id == first.id

    warnings = await _goal_warnings(db_session, orch_goal.id)
    assert len(warnings) == 2
    assert {warning.warning_type for warning in warnings} == {
        "team_hierarchy_skipped",
        "team_hierarchy_not_reviewed",
    }


@pytest.mark.asyncio
async def test_process_skip_transitions_running_process_in_place(db_session, orch_goal):
    # Skipping a *running* current run must convert that row to "skipped"
    # in place, not create a second row while leaving the first stuck in
    # "running" (just marked superseded) — that would misrepresent an
    # abandoned run as still active in its own status column.
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    svc = OrchestrationProcessService()
    running = await svc.start_process(
        db_session, orch_goal.id, process_type="agent_definition_review", trigger_reason="x",
    )
    skipped = await svc.skip_process(
        db_session, orch_goal.id,
        process_type="agent_definition_review",
        skipped_by="human:1234",
        reason="not needed for this goal",
    )
    assert skipped.id == running.id
    assert skipped.status == "skipped"
    assert skipped.superseded_by_id is None

    warnings = await _goal_warnings(db_session, orch_goal.id)
    generic = next(w for w in warnings if w.warning_type == "agent_definition_review_skipped")
    assert generic.source_process_run_id == skipped.id


@pytest.mark.asyncio
async def test_process_skip_rejects_stale_object_after_terminal_transition(db_session, orch_goal):
    from sqlalchemy import update

    from huddleroom.models.orchestration_process import OrchestrationProcessRun
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    svc = OrchestrationProcessService()
    running = await svc.start_process(
        db_session, orch_goal.id, process_type="team_hierarchy", trigger_reason="x",
    )
    await db_session.execute(
        update(OrchestrationProcessRun)
        .where(OrchestrationProcessRun.id == running.id)
        .values(status="completed", outputs={"winner": "other session"}, completed_at=_utcnow())
        .execution_options(synchronize_session=False)
    )

    with pytest.raises(ValueError):
        await svc.skip_process(
            db_session, orch_goal.id,
            process_type="team_hierarchy",
            skipped_by="human:1234",
            reason="stale request",
        )

    await db_session.refresh(running)
    assert running.status == "completed"
    assert running.outputs == {"winner": "other session"}


@pytest.mark.asyncio
async def test_process_complete_rejects_stale_object_after_terminal_transition(db_session, orch_goal):
    from sqlalchemy import update

    from huddleroom.models.orchestration_process import OrchestrationProcessRun
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    svc = OrchestrationProcessService()
    running = await svc.start_process(
        db_session, orch_goal.id, process_type="goal_definition", trigger_reason="x",
    )
    await db_session.execute(
        update(OrchestrationProcessRun)
        .where(OrchestrationProcessRun.id == running.id)
        .values(
            status="skipped",
            skipped_by="human:other",
            override_reason="other session",
            completed_at=_utcnow(),
        )
        .execution_options(synchronize_session=False)
    )

    with pytest.raises(ValueError):
        await svc.complete_process(db_session, running, outputs={"stale": True})

    await db_session.refresh(running)
    assert running.status == "skipped"
    assert running.override_reason == "other session"


@pytest.mark.asyncio
async def test_process_skip_requires_human_attribution(db_session, orch_goal):
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    svc = OrchestrationProcessService()
    with pytest.raises(ValueError):
        await svc.skip_process(
            db_session, orch_goal.id,
            process_type="team_hierarchy",
            skipped_by="manager:42",
            reason="manager thinks it's unnecessary",
        )


@pytest.mark.asyncio
async def test_process_rerun_supersedes_previous(db_session, orch_goal):
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    svc = OrchestrationProcessService()
    first = await svc.start_process(
        db_session, orch_goal.id, process_type="team_hierarchy", trigger_reason="initial",
    )
    await svc.complete_process(db_session, first)

    second = await svc.start_process(
        db_session, orch_goal.id, process_type="team_hierarchy", trigger_reason="human requested re-verify",
    )
    assert first.superseded_by_id == second.id
    assert second.superseded_by_id is None

    current = await svc.get_current(db_session, orch_goal.id, "team_hierarchy")
    assert current.id == second.id


@pytest.mark.asyncio
async def test_process_types_are_independent(db_session, orch_goal):
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    svc = OrchestrationProcessService()
    goal_def = await svc.start_process(
        db_session, orch_goal.id, process_type="goal_definition", trigger_reason="x",
    )
    await svc.start_process(
        db_session, orch_goal.id, process_type="manager_selection", trigger_reason="y",
    )
    # starting manager_selection must not supersede goal_definition
    assert goal_def.superseded_by_id is None
    runs = await svc.list_process_runs(db_session, orch_goal.id)
    assert [r.process_type for r in runs] == ["goal_definition", "manager_selection"]


@pytest.mark.asyncio
async def test_process_unknown_type_rejected(db_session, orch_goal):
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    svc = OrchestrationProcessService()
    with pytest.raises(ValueError):
        await svc.start_process(
            db_session, orch_goal.id, process_type="vibe_check", trigger_reason="x",
        )


@pytest_asyncio.fixture
async def other_goal(db_session, test_project):
    from huddleroom.models.orchestration import OrchestrationGoal

    goal = OrchestrationGoal(
        project_id=test_project.id,
        objective="Unrelated goal",
        success_criteria=[],
    )
    db_session.add(goal)
    await db_session.flush()
    return goal


@pytest_asyncio.fixture
async def other_run(db_session, other_goal):
    from huddleroom.models.orchestration import OrchestrationRun

    run = OrchestrationRun(goal_id=other_goal.id)
    db_session.add(run)
    await db_session.flush()
    return run


@pytest.mark.asyncio
async def test_process_start_rejects_run_from_another_goal(db_session, orch_goal, other_run):
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    svc = OrchestrationProcessService()
    with pytest.raises(ValueError):
        await svc.start_process(
            db_session, orch_goal.id,
            process_type="goal_definition", trigger_reason="x", run_id=other_run.id,
        )


@pytest.mark.asyncio
async def test_process_skip_rejects_run_from_another_goal(db_session, orch_goal, other_run):
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    svc = OrchestrationProcessService()
    with pytest.raises(ValueError):
        await svc.skip_process(
            db_session, orch_goal.id,
            process_type="team_hierarchy", skipped_by="human:1", reason="x", run_id=other_run.id,
        )


@pytest_asyncio.fixture
async def test_agent_2(db_session):
    from huddleroom.models.agent import Agent

    agent = Agent(
        name=f"test-agent-2-{uuid.uuid4()}",
        role="developer",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=[],
        config={},
    )
    db_session.add(agent)
    await db_session.flush()
    return agent


@pytest.mark.asyncio
async def test_warning_service_create_and_links(db_session, orch_goal, orch_run):
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    process_run = await OrchestrationProcessService().start_process(
        db_session, orch_goal.id, process_type="team_hierarchy", trigger_reason="x", run_id=orch_run.id,
    )
    svc = OrchestrationWarningService()
    warning = await svc.create_warning(
        db_session, orch_goal.id,
        warning_type="hierarchy_review_skipped",
        severity="warning",
        message="Team hierarchy review was skipped.",
        run_id=orch_run.id,
        source_process_run_id=process_run.id,
    )
    assert warning.goal_id == orch_goal.id
    assert warning.run_id == orch_run.id
    assert warning.source_process_run_id == process_run.id
    assert bool(warning.active) is True


@pytest.mark.asyncio
async def test_warning_create_is_idempotent(db_session, orch_goal, orch_run):
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    svc = OrchestrationWarningService()
    kwargs = dict(
        warning_type="no_manager", severity="warning", message="No manager selected.",
        run_id=orch_run.id,
    )
    first = await svc.create_warning(db_session, orch_goal.id, **kwargs)
    again = await svc.create_warning(db_session, orch_goal.id, **kwargs)
    assert again.id == first.id
    all_warnings = await svc.list_warnings(db_session, orch_goal.id)
    assert len(all_warnings) == 1


@pytest.mark.asyncio
async def test_warning_service_rejects_bad_severity(db_session, orch_goal):
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    with pytest.raises(ValueError):
        await OrchestrationWarningService().create_warning(
            db_session, orch_goal.id,
            warning_type="x", severity="mild", message="y",
        )


@pytest.mark.asyncio
async def test_warning_acknowledge_keeps_active(db_session, orch_goal):
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    svc = OrchestrationWarningService()
    warning = await svc.create_warning(
        db_session, orch_goal.id,
        warning_type="no_manager", severity="warning", message="No manager selected.",
    )
    acked = await svc.acknowledge_warning(db_session, warning, acknowledged_by="human:1234")
    assert acked.acknowledged_by == "human:1234"
    assert acked.acknowledged_at is not None
    assert bool(acked.active) is True  # acknowledged risk stays visible (spec 11.9)


@pytest.mark.asyncio
async def test_warning_manual_resolve(db_session, orch_goal):
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    svc = OrchestrationWarningService()
    warning = await svc.create_warning(
        db_session, orch_goal.id,
        warning_type="no_manager", severity="warning", message="No manager selected.",
    )
    resolved = await svc.resolve_warning(
        db_session, warning, resolved_by="human:1234", reason="manager assigned manually",
    )
    assert bool(resolved.active) is False
    assert resolved.resolved_by == "human:1234"
    assert resolved.resolved_reason == "manager assigned manually"
    assert resolved.resolved_at is not None


@pytest.mark.asyncio
async def test_warning_auto_resolve(db_session, orch_goal):
    from huddleroom.services.orchestration_warning_service import (
        AUTO_RESOLVED_BY,
        OrchestrationWarningService,
    )

    svc = OrchestrationWarningService()
    warning = await svc.create_warning(
        db_session, orch_goal.id,
        warning_type="no_independent_verifier", severity="warning", message="No verifier.",
    )
    resolved = await svc.resolve_warning(
        db_session, warning,
        resolved_by=AUTO_RESOLVED_BY,
        reason="hierarchy rerun found safe verifier after agent added",
    )
    assert resolved.resolved_by == "system"
    assert resolved.resolved_reason.startswith("hierarchy rerun")
    assert bool(resolved.active) is False


@pytest.mark.asyncio
async def test_warning_resolve_twice_rejected(db_session, orch_goal):
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    svc = OrchestrationWarningService()
    warning = await svc.create_warning(
        db_session, orch_goal.id,
        warning_type="x", severity="recommendation", message="y",
    )
    await svc.resolve_warning(db_session, warning, resolved_by="human:1", reason="done")
    with pytest.raises(ValueError):
        await svc.resolve_warning(db_session, warning, resolved_by="human:1", reason="again")
    with pytest.raises(ValueError):
        await svc.acknowledge_warning(db_session, warning, acknowledged_by="human:1")


@pytest.mark.asyncio
async def test_warning_resolve_concurrent_stale_object_rejected(db_session, orch_goal):
    # A stale in-memory warning object must not be able to overwrite a row
    # that another session already resolved: the update is active-conditional,
    # so it must affect zero rows and raise (Finding 2 & 3).
    from sqlalchemy import update

    from huddleroom.models.orchestration_process import OrchestrationWarning
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    svc = OrchestrationWarningService()
    warning = await svc.create_warning(
        db_session, orch_goal.id,
        warning_type="x", severity="warning", message="y",
    )
    # Simulate concurrent resolution via direct DB manipulation
    await db_session.execute(
        update(OrchestrationWarning)
        .where(OrchestrationWarning.id == warning.id)
        .values(
            active=False,
            resolved_by="human:other",
            resolved_reason="resolved by another session",
            resolved_at=_utcnow(),
        )
        .execution_options(synchronize_session=False)
    )
    # Try to resolve via the stale in-memory object
    with pytest.raises(ValueError) as exc_info:
        await svc.resolve_warning(db_session, warning, resolved_by="human:1", reason="done")
    assert "already resolved" in str(exc_info.value)


@pytest.mark.asyncio
async def test_warning_acknowledge_concurrent_stale_object_rejected(db_session, orch_goal):
    # A stale in-memory warning must not be able to acknowledge a row
    # that was concurrently resolved (Finding 2 & 3).
    from sqlalchemy import update

    from huddleroom.models.orchestration_process import OrchestrationWarning
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    svc = OrchestrationWarningService()
    warning = await svc.create_warning(
        db_session, orch_goal.id,
        warning_type="x", severity="warning", message="y",
    )
    # Simulate concurrent resolution
    await db_session.execute(
        update(OrchestrationWarning)
        .where(OrchestrationWarning.id == warning.id)
        .values(
            active=False,
            resolved_by="human:other",
            resolved_reason="resolved elsewhere",
            resolved_at=_utcnow(),
        )
        .execution_options(synchronize_session=False)
    )
    # Try to acknowledge via the stale in-memory object
    with pytest.raises(ValueError) as exc_info:
        await svc.acknowledge_warning(db_session, warning, acknowledged_by="human:1")
    assert "cannot acknowledge a resolved warning" in str(exc_info.value)


@pytest.mark.asyncio
async def test_warning_resolve_actor_validation_rejects_malformed(db_session, orch_goal):
    # Actor attribution validation must reject malformed values:
    # "human:" (empty suffix), "manager:" (empty suffix), "orchestrator:"
    # (empty suffix), and "orchestratorXYZ" (missing colon for process spec).
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    svc = OrchestrationWarningService()
    warning = await svc.create_warning(
        db_session, orch_goal.id,
        warning_type="x", severity="warning", message="y",
    )
    # Test malformed actor formats
    for bad_actor in ["human:", "manager:", "orchestrator:"]:
        with pytest.raises(ValueError) as exc_info:
            await svc.resolve_warning(db_session, warning, resolved_by=bad_actor, reason="done")
        assert "invalid resolved_by attribution" in str(exc_info.value)
        # Refresh since resolve failed
        await db_session.refresh(warning)

    # orchestratorXYZ (no colon, not a valid format)
    with pytest.raises(ValueError) as exc_info:
        await svc.resolve_warning(db_session, warning, resolved_by="orchestratorXYZ", reason="done")
    assert "invalid resolved_by attribution" in str(exc_info.value)


@pytest.mark.asyncio
async def test_warning_resolve_actor_validation_accepts_valid(db_session, orch_goal):
    # Actor attribution validation must accept valid formats:
    # "orchestrator", "orchestrator:some_process", "human:u1", "manager:agentX", "system"
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    svc = OrchestrationWarningService()
    valid_actors = [
        "orchestrator",
        "orchestrator:baseline_check",
        "human:u1",
        "manager:agentX",
        "system",
    ]
    for valid_actor in valid_actors:
        warning = await svc.create_warning(
            db_session, orch_goal.id,
            warning_type=f"test_{valid_actor}", severity="warning", message="y",
        )
        resolved = await svc.resolve_warning(
            db_session, warning, resolved_by=valid_actor, reason="done"
        )
        assert resolved.resolved_by == valid_actor
        assert resolved.active is False or resolved.active == 0


@pytest.mark.asyncio
async def test_warning_acknowledge_actor_validation_rejects_human_colon_alone(db_session, orch_goal):
    # Acknowledgement must reject "human:" (empty user_id after prefix).
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    svc = OrchestrationWarningService()
    warning = await svc.create_warning(
        db_session, orch_goal.id,
        warning_type="x", severity="warning", message="y",
    )
    with pytest.raises(ValueError) as exc_info:
        await svc.acknowledge_warning(db_session, warning, acknowledged_by="human:")
    assert "must be human attribution" in str(exc_info.value)


@pytest.mark.asyncio
async def test_warning_acknowledge_actor_validation_accepts_valid(db_session, orch_goal):
    # Acknowledgement must accept valid human format: "human:u1", "human:user_123", etc.
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    svc = OrchestrationWarningService()
    warning = await svc.create_warning(
        db_session, orch_goal.id,
        warning_type="x", severity="warning", message="y",
    )
    acked = await svc.acknowledge_warning(db_session, warning, acknowledged_by="human:u1")
    assert acked.acknowledged_by == "human:u1"
    assert acked.active is True or acked.active == 1


@pytest.mark.asyncio
async def test_warning_list_active_only(db_session, orch_goal):
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    svc = OrchestrationWarningService()
    keep = await svc.create_warning(
        db_session, orch_goal.id, warning_type="a", severity="warning", message="active one",
    )
    gone = await svc.create_warning(
        db_session, orch_goal.id, warning_type="b", severity="warning", message="resolved one",
    )
    await svc.resolve_warning(db_session, gone, resolved_by="human:1", reason="fixed")

    all_warnings = await svc.list_warnings(db_session, orch_goal.id)
    active = await svc.list_warnings(db_session, orch_goal.id, active_only=True)
    assert {w.id for w in all_warnings} == {keep.id, gone.id}
    assert [w.id for w in active] == [keep.id]


@pytest.mark.asyncio
async def test_warning_create_does_not_collapse_distinct_related_agents(
    db_session, orch_goal, test_agent, test_agent_2
):
    # Same goal, same warning_type, same (absent) source_process_run_id —
    # but two different agents. Must stay two warnings, not collapse into
    # one (Spec Deviation 10): dedup on goal+type+source alone would lose
    # the second agent's warning entirely.
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    svc = OrchestrationWarningService()
    first = await svc.create_warning(
        db_session, orch_goal.id,
        warning_type="no_independent_verifier", severity="warning", message="agent 1",
        related_agent_id=test_agent.id,
    )
    second = await svc.create_warning(
        db_session, orch_goal.id,
        warning_type="no_independent_verifier", severity="warning", message="agent 2",
        related_agent_id=test_agent_2.id,
    )
    assert first.id != second.id
    all_warnings = await svc.list_warnings(db_session, orch_goal.id)
    assert len(all_warnings) == 2


@pytest.mark.asyncio
async def test_warning_create_rejects_run_from_another_goal(db_session, orch_goal, other_run):
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    with pytest.raises(ValueError):
        await OrchestrationWarningService().create_warning(
            db_session, orch_goal.id,
            warning_type="x", severity="warning", message="y", run_id=other_run.id,
        )


@pytest.mark.asyncio
async def test_warning_create_rejects_gate_from_another_goal(db_session, orch_goal, other_run):
    from huddleroom.models.orchestration import OrchestrationGate
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    other_gate = OrchestrationGate(
        run_id=other_run.id, success_criterion_key="works", gate_type="automated_test",
    )
    db_session.add(other_gate)
    await db_session.flush()

    with pytest.raises(ValueError):
        await OrchestrationWarningService().create_warning(
            db_session, orch_goal.id,
            warning_type="x", severity="warning", message="y", related_gate_id=other_gate.id,
        )


@pytest.mark.asyncio
async def test_warning_create_rejects_action_from_another_goal(db_session, orch_goal, other_run):
    from huddleroom.models.orchestration import OrchestrationAction
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    other_action = OrchestrationAction(
        run_id=other_run.id, idempotency_key="k", action_type="run_command", request={},
    )
    db_session.add(other_action)
    await db_session.flush()

    with pytest.raises(ValueError):
        await OrchestrationWarningService().create_warning(
            db_session, orch_goal.id,
            warning_type="x", severity="warning", message="y", related_action_id=other_action.id,
        )


@pytest.mark.asyncio
async def test_warning_create_rejects_source_process_run_from_another_goal(
    db_session, orch_goal, other_goal
):
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    other_process_run = await OrchestrationProcessService().start_process(
        db_session, other_goal.id, process_type="goal_definition", trigger_reason="x",
    )
    with pytest.raises(ValueError):
        await OrchestrationWarningService().create_warning(
            db_session, orch_goal.id,
            warning_type="x", severity="warning", message="y",
            source_process_run_id=other_process_run.id,
        )


@pytest.mark.asyncio
async def test_warning_create_rejects_source_review_from_another_goal(
    db_session, orch_goal, other_goal, test_agent
):
    from huddleroom.models.orchestration_process import OrchestrationAgentReview
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    other_review = OrchestrationAgentReview(
        goal_id=other_goal.id,
        agent_id=test_agent.id,
        definition_snapshot={"role": "developer"},
        fit_summary="Reviewed for another goal.",
    )
    db_session.add(other_review)
    await db_session.flush()

    with pytest.raises(ValueError, match="source_agent_review.*does not belong to goal"):
        await OrchestrationWarningService().create_warning(
            db_session, orch_goal.id,
            warning_type="x", severity="warning", message="y",
            source_agent_review_id=other_review.id,
        )


@pytest.mark.asyncio
async def test_warning_dedup_identity_includes_source_agent_review(
    db_session, orch_goal, test_agent
):
    from huddleroom.models.orchestration_process import OrchestrationAgentReview
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    reviews = [
        OrchestrationAgentReview(
            goal_id=orch_goal.id,
            agent_id=test_agent.id,
            definition_snapshot={"role": "developer"},
            fit_summary=f"Review {number}.",
        )
        for number in (1, 2)
    ]
    db_session.add_all(reviews)
    await db_session.flush()

    svc = OrchestrationWarningService()
    first = await svc.create_warning(
        db_session, orch_goal.id,
        warning_type="review_risk", severity="warning", message="Review risk.",
        source_agent_review_id=reviews[0].id,
    )
    second = await svc.create_warning(
        db_session, orch_goal.id,
        warning_type="review_risk", severity="warning", message="Review risk.",
        source_agent_review_id=reviews[1].id,
    )
    repeat = await svc.create_warning(
        db_session, orch_goal.id,
        warning_type="review_risk", severity="warning", message="Review risk.",
        source_agent_review_id=reviews[0].id,
    )

    assert first.id != second.id
    assert repeat.id == first.id


@pytest.mark.asyncio
async def test_warning_preserves_history_when_source_review_is_deleted(
    db_session, orch_goal, test_agent
):
    from huddleroom.models.orchestration_process import OrchestrationAgentReview
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    review = OrchestrationAgentReview(
        goal_id=orch_goal.id,
        agent_id=test_agent.id,
        definition_snapshot={"role": "developer"},
        fit_summary="Review with a warning.",
    )
    db_session.add(review)
    await db_session.flush()
    warning = await OrchestrationWarningService().create_warning(
        db_session, orch_goal.id,
        warning_type="review_risk", severity="warning", message="Review risk.",
        source_agent_review_id=review.id,
    )

    await db_session.delete(review)
    await db_session.flush()
    await db_session.refresh(warning)

    assert warning.source_agent_review_id is None
    assert warning.message == "Review risk."


@pytest.mark.asyncio
async def test_decision_create_pending_with_links(db_session, orch_goal, orch_run):
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    process_run = await OrchestrationProcessService().start_process(
        db_session, orch_goal.id, process_type="goal_definition", trigger_reason="x", run_id=orch_run.id,
    )
    svc = OrchestrationAuthorityDecisionService()
    decision = await svc.create_pending(
        db_session, orch_goal.id,
        decision_key="goal_definition:tradeoffs",
        title="Pick primary tradeoff",
        question="Optimize for speed, quality, or cost?",
        authority="human",
        options=["speed", "quality", "cost"],
        recommendation="quality",
        run_id=orch_run.id,
        source_process_run_id=process_run.id,
    )
    assert decision.status == "pending"
    assert decision.goal_id == orch_goal.id
    assert decision.run_id == orch_run.id
    assert decision.source_process_run_id == process_run.id


@pytest.mark.asyncio
async def test_decision_create_pending_is_idempotent(db_session, orch_goal):
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService

    svc = OrchestrationAuthorityDecisionService()
    kwargs = dict(
        decision_key="manager_selection:candidate",
        title="Approve manager",
        question="Approve suggested manager?",
        authority="human",
    )
    first = await svc.create_pending(db_session, orch_goal.id, **kwargs)
    second = await svc.create_pending(db_session, orch_goal.id, **kwargs)
    assert second.id == first.id


@pytest.mark.asyncio
async def test_decision_create_pending_rejects_non_list_options(db_session, orch_goal):
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService

    svc = OrchestrationAuthorityDecisionService()
    with pytest.raises(ValueError):
        await svc.create_pending(
            db_session, orch_goal.id,
            decision_key="k1a", title="t", question="q", authority="human",
            options="yes",
        )
    with pytest.raises(ValueError):
        await svc.create_pending(
            db_session, orch_goal.id,
            decision_key="k1b", title="t", question="q", authority="human",
            options={"key": "yes"},
        )


@pytest.mark.asyncio
async def test_decision_answer_with_selected_option(db_session, orch_goal, test_user):
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService

    svc = OrchestrationAuthorityDecisionService()
    decision = await svc.create_pending(
        db_session, orch_goal.id,
        decision_key="team_hierarchy:verifier",
        title="Continue without independent verifier",
        question="No safe independent verifier found. How to proceed?",
        authority="human",
        options=[
            {"key": "add_verifier", "description": "Add a verifier agent"},
            {"key": "human_verification", "description": "Human verifies"},
            {"key": "continue_with_warning", "description": "Accept the risk"},
        ],
        recommendation="add_verifier",
    )
    answered = await svc.answer_decision(
        db_session, decision,
        selected_option="continue_with_warning",
        reason="budget too small for another agent",
        decided_by_user_id=test_user.id,
    )
    assert answered.status == "answered"
    assert answered.selected_option == "continue_with_warning"
    assert answered.reason == "budget too small for another agent"
    assert answered.decided_by_user_id == test_user.id
    assert bool(answered.overrides_recommendation) is True
    assert answered.decided_at is not None


@pytest.mark.asyncio
async def test_decision_answer_validates_option(db_session, orch_goal, test_user):
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService

    svc = OrchestrationAuthorityDecisionService()
    decision = await svc.create_pending(
        db_session, orch_goal.id,
        decision_key="k1", title="t", question="q", authority="human",
        options=["yes", "no"],
    )
    with pytest.raises(ValueError):
        await svc.answer_decision(
            db_session, decision, selected_option="maybe", decided_by_user_id=test_user.id,
        )


@pytest.mark.asyncio
async def test_decision_answer_rejects_empty_selected_option(db_session, orch_goal, test_user):
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService

    svc = OrchestrationAuthorityDecisionService()
    decision = await svc.create_pending(
        db_session, orch_goal.id,
        decision_key="k1c", title="t", question="q", authority="human",
    )
    with pytest.raises(ValueError):
        await svc.answer_decision(
            db_session, decision, selected_option=None, decided_by_user_id=test_user.id,
        )
    with pytest.raises(ValueError):
        await svc.answer_decision(
            db_session, decision, selected_option="", decided_by_user_id=test_user.id,
        )
    assert decision.status == "pending"


@pytest.mark.asyncio
async def test_decision_answer_requires_exactly_one_decider(db_session, orch_goal, test_user, test_agent):
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService

    svc = OrchestrationAuthorityDecisionService()
    decision = await svc.create_pending(
        db_session, orch_goal.id,
        decision_key="k2", title="t", question="q", authority="manager",
        authority_agent_id=test_agent.id,
        options=["yes", "no"],
    )
    with pytest.raises(ValueError):
        await svc.answer_decision(db_session, decision, selected_option="yes")
    with pytest.raises(ValueError):
        await svc.answer_decision(
            db_session, decision, selected_option="yes",
            decided_by_user_id=test_user.id, decided_by_agent_id=test_agent.id,
        )
    answered = await svc.answer_decision(
        db_session, decision, selected_option="yes", decided_by_agent_id=test_agent.id,
    )
    assert answered.decided_by_agent_id == test_agent.id


@pytest.mark.asyncio
async def test_decision_answer_enforces_declared_authority(db_session, orch_goal, test_user, test_agent):
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService

    svc = OrchestrationAuthorityDecisionService()
    human_decision = await svc.create_pending(
        db_session, orch_goal.id,
        decision_key="k2a", title="t", question="q", authority="human",
        options=["yes", "no"],
    )
    with pytest.raises(ValueError):
        # only the human may answer a human-authority decision (spec 4.3, 6.5)
        await svc.answer_decision(
            db_session, human_decision, selected_option="yes", decided_by_agent_id=test_agent.id,
        )

    manager_decision = await svc.create_pending(
        db_session, orch_goal.id,
        decision_key="k2b", title="t", question="q", authority="manager",
        authority_agent_id=test_agent.id,
        options=["yes", "no"],
    )
    with pytest.raises(ValueError):
        # a manager-authority decision cannot be answered by a human directly
        await svc.answer_decision(
            db_session, manager_decision, selected_option="yes", decided_by_user_id=test_user.id,
        )


@pytest.mark.asyncio
async def test_decision_answer_enforces_specific_authority_agent(
    db_session, orch_goal, test_agent, test_agent_2
):
    # authority="manager" alone only names a role. Two different agents are
    # both plausible "managers" in the system; only the one this decision
    # named via authority_agent_id may answer it (Spec Deviation 9).
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService

    svc = OrchestrationAuthorityDecisionService()
    decision = await svc.create_pending(
        db_session, orch_goal.id,
        decision_key="k2c", title="t", question="q", authority="manager",
        authority_agent_id=test_agent.id,
        options=["yes", "no"],
    )
    with pytest.raises(ValueError):
        await svc.answer_decision(
            db_session, decision, selected_option="yes", decided_by_agent_id=test_agent_2.id,
        )
    answered = await svc.answer_decision(
        db_session, decision, selected_option="yes", decided_by_agent_id=test_agent.id,
    )
    assert answered.decided_by_agent_id == test_agent.id


@pytest.mark.asyncio
async def test_decision_create_pending_requires_authority_agent_for_non_human(db_session, orch_goal):
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService

    svc = OrchestrationAuthorityDecisionService()
    with pytest.raises(ValueError):
        await svc.create_pending(
            db_session, orch_goal.id,
            decision_key="k2d", title="t", question="q", authority="manager",
        )


@pytest.mark.asyncio
async def test_decision_create_pending_rejects_authority_agent_for_human(db_session, orch_goal, test_agent):
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService

    svc = OrchestrationAuthorityDecisionService()
    with pytest.raises(ValueError):
        await svc.create_pending(
            db_session, orch_goal.id,
            decision_key="k2e", title="t", question="q", authority="human",
            authority_agent_id=test_agent.id,
        )


@pytest.mark.asyncio
async def test_decision_stores_consequences(db_session, orch_goal, test_user):
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService

    svc = OrchestrationAuthorityDecisionService()
    decision = await svc.create_pending(
        db_session, orch_goal.id,
        decision_key="k2f", title="t", question="q", authority="human",
        options=["add_verifier", "continue_with_warning"],
        consequences="Skipping verification leaves the risk of undetected regressions.",
    )
    assert decision.consequences == "Skipping verification leaves the risk of undetected regressions."
    answered = await svc.answer_decision(
        db_session, decision, selected_option="continue_with_warning",
        decided_by_user_id=test_user.id,
    )
    assert answered.consequences == "Skipping verification leaves the risk of undetected regressions."


@pytest.mark.asyncio
async def test_decision_create_pending_rejects_run_from_another_goal(db_session, orch_goal, other_run):
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService

    with pytest.raises(ValueError):
        await OrchestrationAuthorityDecisionService().create_pending(
            db_session, orch_goal.id,
            decision_key="k2g", title="t", question="q", authority="human",
            run_id=other_run.id,
        )


@pytest.mark.asyncio
async def test_decision_create_pending_rejects_source_process_run_from_another_goal(
    db_session, orch_goal, other_goal
):
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    other_process_run = await OrchestrationProcessService().start_process(
        db_session, other_goal.id, process_type="goal_definition", trigger_reason="x",
    )
    with pytest.raises(ValueError):
        await OrchestrationAuthorityDecisionService().create_pending(
            db_session, orch_goal.id,
            decision_key="k2h", title="t", question="q", authority="human",
            source_process_run_id=other_process_run.id,
        )


@pytest.mark.asyncio
async def test_decision_create_pending_fresh_after_terminal(db_session, orch_goal, test_user):
    # A rerun must be able to raise a fresh decision under the same logical
    # key once the prior one went terminal (spec 6.3) — create_pending's
    # idempotency only matches an existing *pending* row for the key.
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService

    svc = OrchestrationAuthorityDecisionService()
    kwargs = dict(
        decision_key="team_hierarchy:verifier2", title="t", question="q",
        authority="human", options=["yes", "no"],
    )
    first = await svc.create_pending(db_session, orch_goal.id, **kwargs)
    await svc.answer_decision(
        db_session, first, selected_option="yes", decided_by_user_id=test_user.id,
    )
    second = await svc.create_pending(db_session, orch_goal.id, **kwargs)
    assert second.id != first.id
    assert second.status == "pending"


@pytest.mark.asyncio
async def test_decision_answer_requires_pending(db_session, orch_goal, test_user):
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService

    svc = OrchestrationAuthorityDecisionService()
    decision = await svc.create_pending(
        db_session, orch_goal.id,
        decision_key="k3", title="t", question="q", authority="human",
        options=["yes", "no"],
    )
    await svc.answer_decision(
        db_session, decision, selected_option="yes", decided_by_user_id=test_user.id,
    )
    with pytest.raises(ValueError):
        await svc.answer_decision(
            db_session, decision, selected_option="no", decided_by_user_id=test_user.id,
        )


@pytest.mark.asyncio
async def test_decision_cancel_pending(db_session, orch_goal):
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService

    svc = OrchestrationAuthorityDecisionService()
    decision = await svc.create_pending(
        db_session, orch_goal.id,
        decision_key="k4", title="t", question="q", authority="human",
    )
    cancelled = await svc.cancel_decision(db_session, decision, reason="goal re-scoped")
    assert cancelled.status == "cancelled"
    assert cancelled.reason == "goal re-scoped"
    assert cancelled.decided_at is None
    with pytest.raises(ValueError):
        await svc.cancel_decision(db_session, decision, reason="again")


@pytest.mark.asyncio
async def test_decision_list_filter_by_status(db_session, orch_goal, test_user):
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService

    svc = OrchestrationAuthorityDecisionService()
    pending = await svc.create_pending(
        db_session, orch_goal.id, decision_key="k5", title="t", question="q", authority="human",
    )
    answered = await svc.create_pending(
        db_session, orch_goal.id, decision_key="k6", title="t", question="q", authority="human",
        options=["yes"],
    )
    await svc.answer_decision(
        db_session, answered, selected_option="yes", decided_by_user_id=test_user.id,
    )
    all_rows = await svc.list_decisions(db_session, orch_goal.id)
    pending_rows = await svc.list_decisions(db_session, orch_goal.id, status="pending")
    assert {d.id for d in all_rows} == {pending.id, answered.id}
    assert [d.id for d in pending_rows] == [pending.id]


@pytest.mark.asyncio
async def test_decision_create_pending_rejects_malformed_options(db_session, orch_goal):
    # A dict option without "key", or a non-str/dict entry, must be rejected
    # at creation time — silently dropping it would leave answer_decision's
    # membership check with an empty key set, which must never mean "any
    # selected_option is accepted" (spec 10.4: options are code-validated).
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService

    svc = OrchestrationAuthorityDecisionService()
    with pytest.raises(ValueError):
        await svc.create_pending(
            db_session, orch_goal.id,
            decision_key="k8", title="t", question="q", authority="human",
            options=[{"description": "missing key"}],
        )
    with pytest.raises(ValueError):
        await svc.create_pending(
            db_session, orch_goal.id,
            decision_key="k9", title="t", question="q", authority="human",
            options=[123],
        )
    with pytest.raises(ValueError):
        await svc.create_pending(
            db_session, orch_goal.id,
            decision_key="k10", title="t", question="q", authority="human",
            options=["yes", "yes"],  # duplicate key
        )


@pytest.mark.asyncio
async def test_decision_rejects_bad_authority(db_session, orch_goal):
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService

    svc = OrchestrationAuthorityDecisionService()
    with pytest.raises(ValueError):
        await svc.create_pending(
            db_session, orch_goal.id,
            decision_key="k7", title="t", question="q", authority="committee",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_options", [{}, "", (), 0, False])
async def test_decision_create_pending_rejects_falsey_non_list_options(db_session, orch_goal, bad_options):
    # {}, "", (), 0, False are all falsey but not None; must be rejected as
    # malformed rather than treated as "no options offered" (which would
    # leave answer_decision's membership check accepting anything).
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService

    with pytest.raises(ValueError):
        await OrchestrationAuthorityDecisionService().create_pending(
            db_session, orch_goal.id,
            decision_key=f"falsey-{bad_options!r}", title="t", question="q", authority="human",
            options=bad_options,
        )


@pytest.mark.asyncio
async def test_decision_create_pending_rejects_gate_from_another_goal(db_session, orch_goal, other_run):
    from huddleroom.models.orchestration import OrchestrationGate
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService

    other_gate = OrchestrationGate(
        run_id=other_run.id, success_criterion_key="works", gate_type="automated_test",
    )
    db_session.add(other_gate)
    await db_session.flush()

    with pytest.raises(ValueError):
        await OrchestrationAuthorityDecisionService().create_pending(
            db_session, orch_goal.id,
            decision_key="k2i", title="t", question="q", authority="human",
            related_gate_id=other_gate.id,
        )


@pytest.mark.asyncio
async def test_decision_create_pending_rejects_action_from_another_goal(db_session, orch_goal, other_run):
    from huddleroom.models.orchestration import OrchestrationAction
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService

    other_action = OrchestrationAction(
        run_id=other_run.id, idempotency_key="k", action_type="run_command", request={},
    )
    db_session.add(other_action)
    await db_session.flush()

    with pytest.raises(ValueError):
        await OrchestrationAuthorityDecisionService().create_pending(
            db_session, orch_goal.id,
            decision_key="k2j", title="t", question="q", authority="human",
            related_action_id=other_action.id,
        )


@pytest.mark.asyncio
async def test_decision_answer_rejects_concurrent_transition(db_session, orch_goal, test_user):
    # A stale in-memory decision object must not be able to overwrite a row
    # that another session already moved to a terminal status: the update
    # is status-conditional, so it must affect zero rows and raise.
    from sqlalchemy import update

    from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService

    svc = OrchestrationAuthorityDecisionService()
    decision = await svc.create_pending(
        db_session, orch_goal.id,
        decision_key="k-race-answer", title="t", question="q", authority="human",
        options=["yes", "no"],
    )
    # synchronize_session=False keeps the in-memory `decision` object
    # reporting status="pending", as if a concurrent session made this
    # change without this session knowing yet.
    await db_session.execute(
        update(OrchestrationAuthorityDecision)
        .where(OrchestrationAuthorityDecision.id == decision.id)
        .values(status="cancelled", reason="raced")
        .execution_options(synchronize_session=False)
    )
    with pytest.raises(ValueError):
        await svc.answer_decision(
            db_session, decision, selected_option="yes", decided_by_user_id=test_user.id,
        )


@pytest.mark.asyncio
async def test_decision_cancel_rejects_concurrent_transition(db_session, orch_goal, test_user):
    from sqlalchemy import update

    from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService

    svc = OrchestrationAuthorityDecisionService()
    decision = await svc.create_pending(
        db_session, orch_goal.id,
        decision_key="k-race-cancel", title="t", question="q", authority="human",
        options=["yes", "no"],
    )
    await db_session.execute(
        update(OrchestrationAuthorityDecision)
        .where(OrchestrationAuthorityDecision.id == decision.id)
        .values(
            status="answered", selected_option="yes", decided_by_user_id=test_user.id,
        )
        .execution_options(synchronize_session=False)
    )
    with pytest.raises(ValueError):
        await svc.cancel_decision(db_session, decision, reason="too late")


@pytest.mark.asyncio
async def test_decision_cancel_rejects_blank_reason(db_session, orch_goal):
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService

    svc = OrchestrationAuthorityDecisionService()
    decision = await svc.create_pending(
        db_session, orch_goal.id, decision_key="k-cancel-blank", title="t", question="q", authority="human",
    )
    with pytest.raises(ValueError):
        await svc.cancel_decision(db_session, decision, reason="")
    with pytest.raises(ValueError):
        await svc.cancel_decision(db_session, decision, reason="   ")


@pytest.mark.asyncio
async def test_process_skip_rejects_blank_reason(db_session, orch_goal):
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    svc = OrchestrationProcessService()
    with pytest.raises(ValueError):
        await svc.skip_process(
            db_session, orch_goal.id, process_type="goal_definition",
            skipped_by="human:1", reason="",
        )
    with pytest.raises(ValueError):
        await svc.skip_process(
            db_session, orch_goal.id, process_type="goal_definition",
            skipped_by="human:1", reason="   ",
        )


@pytest.mark.asyncio
async def test_warning_resolve_rejects_blank_reason_and_bad_actor(db_session, orch_goal):
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    svc = OrchestrationWarningService()
    warning = await svc.create_warning(
        db_session, orch_goal.id, warning_type="x", severity="warning", message="y",
    )
    with pytest.raises(ValueError):
        await svc.resolve_warning(db_session, warning, resolved_by="human:1", reason="")
    with pytest.raises(ValueError):
        await svc.resolve_warning(db_session, warning, resolved_by="", reason="done")
    with pytest.raises(ValueError):
        await svc.resolve_warning(db_session, warning, resolved_by="bogus", reason="done")


@pytest.mark.asyncio
async def test_warning_acknowledge_rejects_non_human_attribution(db_session, orch_goal):
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    svc = OrchestrationWarningService()
    warning = await svc.create_warning(
        db_session, orch_goal.id, warning_type="x", severity="warning", message="y",
    )
    with pytest.raises(ValueError):
        await svc.acknowledge_warning(db_session, warning, acknowledged_by="system")
    with pytest.raises(ValueError):
        await svc.acknowledge_warning(db_session, warning, acknowledged_by="manager:1234")
    with pytest.raises(ValueError):
        await svc.acknowledge_warning(db_session, warning, acknowledged_by="")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "differ_field", ["run_id", "source_process_run_id", "related_gate_id", "related_action_id"]
)
async def test_warning_dedup_identity_per_linkage(db_session, orch_goal, differ_field):
    # Every linkage column is part of the dedup match (Spec Deviation 10):
    # two warnings that differ in only one linkage column must stay two
    # distinct rows, and repeating the exact same linkage set must collapse
    # back onto the first (no duplicate active warning for one retry).
    from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGate, OrchestrationRun
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    run_a = OrchestrationRun(goal_id=orch_goal.id)
    # Only one *active* run per goal (partial unique index) — mark run_b
    # terminal so both can coexist under the same goal.
    run_b = OrchestrationRun(goal_id=orch_goal.id, status="completed")
    db_session.add_all([run_a, run_b])
    await db_session.flush()

    process_a = await OrchestrationProcessService().start_process(
        db_session, orch_goal.id, process_type="goal_definition", trigger_reason="x", run_id=run_a.id,
    )
    process_b = await OrchestrationProcessService().start_process(
        db_session, orch_goal.id, process_type="agent_definition_review", trigger_reason="x", run_id=run_a.id,
    )
    gate_a = OrchestrationGate(run_id=run_a.id, success_criterion_key="a", gate_type="automated_test")
    gate_b = OrchestrationGate(run_id=run_a.id, success_criterion_key="b", gate_type="automated_test")
    action_a = OrchestrationAction(run_id=run_a.id, idempotency_key="a", action_type="run_command", request={})
    action_b = OrchestrationAction(run_id=run_a.id, idempotency_key="b", action_type="run_command", request={})
    db_session.add_all([gate_a, gate_b, action_a, action_b])
    await db_session.flush()

    values = {
        "run_id": (run_a.id, run_b.id),
        "source_process_run_id": (process_a.id, process_b.id),
        "related_gate_id": (gate_a.id, gate_b.id),
        "related_action_id": (action_a.id, action_b.id),
    }

    svc = OrchestrationWarningService()
    base_kwargs = dict(warning_type="dedup_check", severity="warning", message="m", run_id=run_a.id)
    base_kwargs[differ_field] = values[differ_field][0]
    variant_kwargs = dict(base_kwargs)
    variant_kwargs[differ_field] = values[differ_field][1]

    first = await svc.create_warning(db_session, orch_goal.id, **base_kwargs)
    second = await svc.create_warning(db_session, orch_goal.id, **variant_kwargs)
    assert first.id != second.id

    repeat = await svc.create_warning(db_session, orch_goal.id, **base_kwargs)
    assert repeat.id == first.id


@pytest.mark.asyncio
async def test_process_skip_transitions_waiting_decision_process_in_place(db_session, orch_goal):
    # Skipping a process in waiting_decision status (parked process) must
    # convert that row to "skipped" in place, not create a second row while
    # leaving the first stuck in "waiting_decision" (Finding 1).
    from sqlalchemy import update

    from huddleroom.models.orchestration_process import OrchestrationProcessRun
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    svc = OrchestrationProcessService()
    waiting = await svc.start_process(
        db_session, orch_goal.id, process_type="team_hierarchy", trigger_reason="x",
    )
    # Simulate a parked process by manually setting status to waiting_decision
    await db_session.execute(
        update(OrchestrationProcessRun)
        .where(OrchestrationProcessRun.id == waiting.id)
        .values(status="waiting_decision")
    )
    await db_session.flush()

    skipped = await svc.skip_process(
        db_session, orch_goal.id,
        process_type="team_hierarchy",
        skipped_by="human:1234",
        reason="not needed after all",
    )
    assert skipped.id == waiting.id
    assert skipped.status == "skipped"
    assert skipped.superseded_by_id is None

    warnings = await _goal_warnings(db_session, orch_goal.id)
    assert {warning.warning_type for warning in warnings} == {
        "team_hierarchy_skipped",
        "team_hierarchy_not_reviewed",
    }
    assert all(warning.source_process_run_id == skipped.id for warning in warnings)


@pytest.mark.asyncio
async def test_decision_answer_rejects_whitespace_only_selected_option(db_session, orch_goal, test_user):
    # A whitespace-only selected_option like "   " must be rejected
    # even on a pending decision with no options (Finding 3).
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService

    svc = OrchestrationAuthorityDecisionService()
    decision = await svc.create_pending(
        db_session, orch_goal.id,
        decision_key="k-whitespace-option", title="t", question="q", authority="human",
        options=[],  # No options offered
    )
    with pytest.raises(ValueError):
        await svc.answer_decision(
            db_session, decision, selected_option="   ", decided_by_user_id=test_user.id,
        )
    assert decision.status == "pending"


@pytest.mark.asyncio
async def test_decision_answer_long_selected_option_persists(db_session, orch_goal, test_user):
    # Free-text goal-definition decisions allow open-ended answers. Verify that
    # selected_option can store strings longer than 255 characters without truncation.
    from sqlalchemy import select

    from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService

    svc = OrchestrationAuthorityDecisionService()
    decision = await svc.create_pending(
        db_session, orch_goal.id,
        decision_key="goal_definition:clarification",
        title="Clarify project scope",
        question="What is the detailed scope?",
        authority="human",
        options=[],  # No bounded options; free-text answer expected
    )
    # Create a long answer string (300+ chars to exceed the old VARCHAR(255) limit)
    long_answer = "The project scope includes: " + "x" * 300
    assert len(long_answer) > 255

    answered = await svc.answer_decision(
        db_session, decision, selected_option=long_answer, decided_by_user_id=test_user.id,
    )
    assert answered.selected_option == long_answer
    assert answered.status == "answered"

    # Verify it round-trips from the DB without truncation
    result = await db_session.execute(
        select(OrchestrationAuthorityDecision).where(OrchestrationAuthorityDecision.id == answered.id)
    )
    fetched = result.scalar_one()
    assert fetched.selected_option == long_answer
    assert len(fetched.selected_option) > 255
