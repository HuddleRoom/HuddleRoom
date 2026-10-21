import asyncio
import json
import uuid
from contextlib import asynccontextmanager

import pytest
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine

from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGoal, OrchestrationRun
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
from huddleroom.models.orchestration_memory import OrchestrationMemorySection
from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
from huddleroom.services.orchestration_process_service import OrchestrationProcessService
from tests.conftest import adopt_session


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
async def db_session(owned_session_factory):
    """Autobegin session (no session.begin()) so the test task can own it."""
    async with owned_session_factory() as db:
        yield db
        await db.rollback()


@pytest.fixture
async def completion_ready_goal(db_session, test_project):
    from huddleroom.models.agent import Agent
    from huddleroom.models.orchestration import OrchestrationEvidence, OrchestrationGate
    from huddleroom.models.session import Session
    from huddleroom.models.task import Task
    from huddleroom.schemas.orchestration import OrchestrationGoalCreate
    from huddleroom.services.orchestration_service import OrchestrationService
    from tests.conftest import complete_baseline_processes

    service = OrchestrationService()
    goal, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Close the goal",
            success_criteria=[{"key": "done", "description": "Work is complete"}],
            constraints={},
            budget={},
        ),
        created_by_user_id=None,
    )
    writer = Agent(
        name="closeout-writer",
        role="writer",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=["summarization"],
        config={},
        is_active=True,
    )
    db_session.add(writer)
    await db_session.flush()
    run.baseline_authorized = True
    await complete_baseline_processes(db_session, goal, run)
    goal.weight = "standard"
    gate = OrchestrationGate(
        run_id=run.id,
        success_criterion_key="plan_item:done",
        gate_type="work_completed",
        required_evidence={"success_criterion_keys": ["done"]},
        status="accepted",
    )
    db_session.add(gate)
    await db_session.flush()
    evidence = OrchestrationEvidence(
        run_id=run.id,
        gate_id=gate.id,
        source_type="verification",
        source_id=goal.id,
        verdict="accepted",
        evidence_metadata={},
    )
    db_session.add(evidence)
    await db_session.flush()
    final_gate = OrchestrationGate(
        run_id=run.id,
        success_criterion_key="final_summary",
        gate_type="final_summary_accepted",
        required_evidence={},
        status="accepted",
    )
    db_session.add(final_gate)
    await db_session.flush()
    task = Task(
        project_id=test_project.id,
        title="Summarize closeout",
        status="done",
        assigned_to=writer.id,
        metadata_={
            "orchestration": {
                "run_id": str(run.id),
                "gate_id": str(final_gate.id),
                "work_function": "summarization",
                "final_summary": True,
            }
        },
    )
    db_session.add(task)
    await db_session.flush()
    session = Session(
        project_id=test_project.id,
        task_id=task.id,
        agent_id=writer.id,
        adapter_type="api",
        status="completed",
        output=json.dumps(
            {
                "summary": "Work is complete.",
                "criteria": [{"criterion_key": "done", "evidence_ids": [str(evidence.id)]}],
                "unresolved_gaps": [],
            }
        ),
        metadata_={},
        origin="auto",
    )
    db_session.add(session)
    await db_session.flush()
    final_evidence = OrchestrationEvidence(
        run_id=run.id,
        gate_id=final_gate.id,
        source_type="session",
        source_id=session.id,
        producer_agent_id=writer.id,
        verdict="accepted",
        evidence_metadata={},
    )
    db_session.add(final_evidence)
    await db_session.flush()

    async def baseline_ready(*_args, **_kwargs):
        return None

    service._ensure_baseline_processes_ready = baseline_ready

    # This fixture drives the goal through closeout/completion, which tick()
    # gates on run.phase == "authorized". Authorize directly so the fixture
    # can exercise completion without invoking the baseline start flow.
    run.phase = "authorized"
    await db_session.flush()

    return service, goal, run


@pytest.fixture
async def trivial_completion_ready_goal(completion_ready_goal):
    service, goal, run = completion_ready_goal
    goal.weight = "trivial"
    return service, goal, run


async def action_for_key(db_session, run_id, idempotency_key):
    from huddleroom.models.orchestration import OrchestrationAction

    return await db_session.scalar(
        select(OrchestrationAction).where(
            OrchestrationAction.run_id == run_id,
            OrchestrationAction.idempotency_key == idempotency_key,
        )
    )


async def stub_tick_baseline(monkeypatch):
    from huddleroom.services import orchestration_service

    async def complete(*_args, **_kwargs):
        return {"status": "completed"}

    for process in (
        orchestration_service.GoalDefinitionProcess,
        orchestration_service.ManagerSelectionProcess,
        orchestration_service.AgentDefinitionReviewProcess,
        orchestration_service.TeamHierarchyProcess,
        orchestration_service.EffectivenessReviewProcess,
    ):
        monkeypatch.setattr(process, "advance", complete)


async def create_goal(db_session, test_project, *, weight: str):
    goal = OrchestrationGoal(
        project_id=test_project.id,
        objective="Close the goal",
        success_criteria=[{"key": "done", "description": "Work is complete"}],
        constraints={},
        budget={},
        weight=weight,
    )
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id)
    db_session.add(run)
    await db_session.flush()
    return goal, run


def completion_manifest_fixture(run):
    return {
        "declared_success_criteria": [{"key": "done", "description": "Work is complete"}],
        "criterion_evidence": [{"key": "done", "status": "accepted"}],
        "accepted_non_summary_gates": [],
        "final_summary": {"run_id": str(run.id), "summary": "Complete"},
        "warning_disposition": [],
        "overridden_gates": [],
    }


async def pending_decisions(db_session, goal_id):
    return await OrchestrationAuthorityDecisionService().list_decisions(
        db_session, goal_id, status="pending"
    )


@pytest.mark.asyncio
async def test_complete_run_action_is_blocked_until_closeout_authorizes(
    db_session, completion_ready_goal
):
    service, _goal, run = completion_ready_goal
    key = f"run:{run.id}:kind:complete_run"

    with pytest.raises(HTTPException) as exc:
        await service.execute_complete_run_action(
            db_session,
            run_id=run.id,
            request={"action_type": "complete_run", "reason": "ready"},
            idempotency_key=key,
        )

    assert exc.value.status_code == 409
    assert exc.value.detail == "Goal closeout must authorize completion"
    assert run.status != "completed"
    assert await action_for_key(db_session, run.id, key) is None


@pytest.mark.asyncio
async def test_standard_tick_parks_closeout_before_complete_run(
    db_session, completion_ready_goal, monkeypatch
):
    adopt_session(db_session)
    service, _goal, run = completion_ready_goal
    await stub_tick_baseline(monkeypatch)

    # End this test-owned setup before the real orchestration operation.
    await db_session.flush()
    await db_session.commit()
    result = await service.tick(db_session, run.id)

    assert result["goal_closeout_process"]["status"] == "waiting_decision"
    assert result["goal_closeout_process"]["decision_id"] is not None
    assert result["run_completed"] is False
    assert result["completion_action_id"] is None


@pytest.mark.asyncio
async def test_approved_closeout_completes_run_in_same_tick(
    db_session, completion_ready_goal, test_user, monkeypatch
):
    adopt_session(db_session)
    service, _goal, run = completion_ready_goal
    await stub_tick_baseline(monkeypatch)
    # End this test-owned setup before the real orchestration operation.
    await db_session.flush()
    await db_session.commit()
    first = await service.tick(db_session, run.id)
    decision = await db_session.get(
        OrchestrationAuthorityDecision,
        uuid.UUID(first["goal_closeout_process"]["decision_id"]),
    )
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session,
        decision,
        selected_option="approve_completion",
        reason="Evidence is sufficient.",
        decided_by_user_id=test_user.id,
    )

    # End this test-owned setup before the real orchestration operation.
    await db_session.flush()
    await db_session.commit()
    result = await service.tick(db_session, run.id)

    assert result["goal_closeout_process"]["status"] == "completed"
    assert result["goal_closeout_process"]["completion_authorized"] is True
    assert result["run_completed"] is True
    assert result["completion_action_id"] is not None


@pytest.mark.asyncio
async def test_trivial_closeout_and_completion_finish_in_one_tick(
    db_session, trivial_completion_ready_goal, monkeypatch
):
    adopt_session(db_session)
    service, _goal, run = trivial_completion_ready_goal
    await stub_tick_baseline(monkeypatch)

    # End this test-owned setup before the real orchestration operation.
    await db_session.flush()
    await db_session.commit()
    result = await service.tick(db_session, run.id)

    assert result["goal_closeout_process"]["full_closeout"] is False
    assert result["run_completed"] is True


async def memory_section(db_session, goal, section_key):
    return await db_session.scalar(
        select(OrchestrationMemorySection).where(
            OrchestrationMemorySection.goal_id == goal.id,
            OrchestrationMemorySection.section_key == section_key,
        )
    )


@pytest.mark.asyncio
async def test_closeout_is_idle_until_completion_preconditions_exist(db_session, test_project):
    from huddleroom.services.orchestration_goal_closeout import GoalCloseoutProcess

    goal, run = await create_goal(db_session, test_project, weight="standard")

    result = await GoalCloseoutProcess().advance(db_session, goal, run, preconditions=None)

    assert result == {
        "status": "idle",
        "mode": None,
        "completion_authorized": False,
    }


@pytest.mark.asyncio
async def test_force_started_closeout_waits_for_preconditions(db_session, test_project):
    from huddleroom.services.orchestration_goal_closeout import GoalCloseoutProcess

    goal, run = await create_goal(db_session, test_project, weight="standard")
    process = await OrchestrationProcessService().start_process(
        db_session,
        goal.id,
        process_type="goal_closeout",
        trigger_reason="human requested: full closeout",
        run_id=run.id,
        input_snapshot={"mode": "completion", "full_closeout": True},
    )

    result = await GoalCloseoutProcess().advance(db_session, goal, run, preconditions=None)

    assert result["status"] == "running"
    assert result["awaiting_preconditions"] is True
    assert process.status == "running"


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_status", ["completed", "skipped"])
async def test_historical_terminal_closeout_neither_authorizes_nor_advances_until_ready(
    db_session, test_project, terminal_status
):
    from huddleroom.services.orchestration_goal_closeout import GoalCloseoutProcess

    goal, historical_run = await create_goal(db_session, test_project, weight="standard")
    process_service = OrchestrationProcessService()
    if terminal_status == "completed":
        historical = await process_service.start_process(
            db_session,
            goal.id,
            process_type="goal_closeout",
            trigger_reason="automatic: prior run completion requested",
            run_id=historical_run.id,
            input_snapshot={"mode": "completion"},
        )
        await process_service.complete_process(
            db_session,
            historical,
            outputs={
                "mode": "completion",
                "completion_authorized": False,
                "gates": {"closeout_completed": False},
            },
        )
    else:
        historical = await process_service.skip_process(
            db_session,
            goal.id,
            process_type="goal_closeout",
            skipped_by="human:test",
            reason="Prior run forced completion.",
            run_id=historical_run.id,
        )
    historical_run.status = "completed"
    await db_session.flush()
    current_run = OrchestrationRun(goal_id=goal.id)
    db_session.add(current_run)
    await db_session.flush()

    with pytest.raises(HTTPException) as exc_info:
        await GoalCloseoutProcess().completion_authorization(db_session, goal, current_run)
    idle = await GoalCloseoutProcess().advance(db_session, goal, current_run, preconditions=None)

    assert exc_info.value.status_code == 409
    assert idle == {"status": "idle", "mode": None, "completion_authorized": False}
    assert historical.superseded_by_id is None

    ready = await GoalCloseoutProcess().advance(
        db_session,
        goal,
        current_run,
        preconditions=completion_manifest_fixture(current_run),
    )
    successor = await process_service.get_current(db_session, goal.id, "goal_closeout")

    assert ready["status"] == "waiting_decision"
    assert successor.run_id == current_run.id
    assert historical.superseded_by_id == successor.id


@pytest.mark.asyncio
async def test_skipped_closeout_requires_warning_ack_before_force_completion(
    db_session, completion_ready_goal, test_user
):
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    service, goal, run = completion_ready_goal
    process_service = OrchestrationProcessService()
    skipped = await process_service.skip_process(
        db_session,
        goal.id,
        process_type="goal_closeout",
        skipped_by=f"human:{test_user.id}",
        reason="Emergency delivery.",
        run_id=run.id,
    )
    warnings = await OrchestrationWarningService().list_warnings(
        db_session, goal.id, active_only=True
    )
    warning = next(
        (
            warning
            for warning in warnings
            if warning.warning_type == "goal_closeout_skipped"
            and warning.source_process_run_id == skipped.id
        ),
        None,
    )

    assert warning is not None
    assert warning.warning_type == "goal_closeout_skipped"
    assert warning.message == (
        "Goal was completed without closeout. No completion rationale or "
        "lessons learned were recorded for this goal. Reason: Emergency delivery."
    )
    assert await service._run_ready_for_completion(db_session, goal, run) is False

    await OrchestrationWarningService().acknowledge_warning(
        db_session,
        warning,
        acknowledged_by=f"human:{test_user.id}",
    )

    assert await service._run_ready_for_completion(db_session, goal, run) is True
    assert skipped.status == "skipped"


@pytest.mark.asyncio
async def test_force_start_after_declined_closeout_creates_successor(db_session, test_project):
    goal, run = await create_goal(db_session, test_project, weight="standard")
    process_service = OrchestrationProcessService()
    declined = await process_service.start_process(
        db_session,
        goal.id,
        process_type="goal_closeout",
        trigger_reason="automatic: completion requested",
        run_id=run.id,
    )
    await process_service.complete_process(
        db_session,
        declined,
        outputs={
            "mode": "completion",
            "completion_authorized": False,
            "gates": {"closeout_completed": False},
        },
    )

    successor = await process_service.start_process(
        db_session,
        goal.id,
        process_type="goal_closeout",
        trigger_reason="human requested: retry closeout",
        run_id=run.id,
        input_snapshot={"mode": "completion", "full_closeout": True},
    )

    assert successor.superseded_by_id is None
    assert declined.superseded_by_id == successor.id
    assert successor.process_type == "goal_closeout"


@pytest.mark.asyncio
async def test_force_start_api_marks_goal_closeout_full(
    client, db_session, completion_ready_goal, auth_headers
):
    _service, goal, run = completion_ready_goal

    response = await client.post(
        f"/api/v1/projects/{goal.project_id}/orchestration/goals/"
        f"{goal.id}/processes/goal_closeout/start",
        json={"reason": "human wants the full closeout"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    process = await OrchestrationProcessService().get_current(
        db_session, goal.id, "goal_closeout"
    )
    assert process.run_id == run.id
    assert process.input_snapshot["full_closeout"] is True


@pytest.mark.asyncio
async def test_authorized_closeout_successor_resolves_prior_skip_warning(
    db_session, test_project, test_user
):
    from huddleroom.services.orchestration_goal_closeout import GoalCloseoutProcess
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    goal, run = await create_goal(db_session, test_project, weight="trivial")
    process_service = OrchestrationProcessService()
    skipped = await process_service.skip_process(
        db_session,
        goal.id,
        process_type="goal_closeout",
        skipped_by=f"human:{test_user.id}",
        reason="Closeout deferred.",
        run_id=run.id,
    )
    warning = (await OrchestrationWarningService().list_warnings(
        db_session, goal.id, active_only=True
    ))[0]
    await OrchestrationWarningService().acknowledge_warning(
        db_session, warning, acknowledged_by=f"human:{test_user.id}"
    )
    successor = await process_service.start_process(
        db_session,
        goal.id,
        process_type="goal_closeout",
        trigger_reason="human requested: retry closeout",
        run_id=run.id,
        input_snapshot={"mode": "completion", "full_closeout": True},
    )

    assert warning.active is True
    assert skipped.superseded_by_id == successor.id

    closeout = GoalCloseoutProcess()
    manifest = completion_manifest_fixture(run)
    await closeout.advance(db_session, goal, run, preconditions=manifest)
    decision = (await pending_decisions(db_session, goal.id))[0]
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session,
        decision,
        selected_option="approve_completion",
        reason="Closeout is complete.",
        decided_by_user_id=test_user.id,
    )
    await closeout.advance(db_session, goal, run, preconditions=manifest)

    assert warning.active is False
    assert warning.resolved_by == "orchestrator:process_rerun"
    assert warning.resolved_reason == "process rerun fixed the skipped-process warning"


@pytest.mark.asyncio
async def test_terminal_tick_returns_skipped_closeout_summary(db_session, test_project):
    from huddleroom.services.orchestration_service import OrchestrationService

    goal, run = await create_goal(db_session, test_project, weight="standard")
    await OrchestrationProcessService().skip_process(
        db_session,
        goal.id,
        process_type="goal_closeout",
        skipped_by="human:test",
        reason="Closeout is not needed.",
        run_id=run.id,
    )
    goal.status = "completed"
    run.status = "completed"
    await db_session.flush()

    # End this test-owned setup before the real orchestration operation.
    await db_session.flush()
    await db_session.commit()
    result = await OrchestrationService().tick(db_session, run.id)

    assert result["goal_closeout_process"] == {
        "status": "skipped",
        "mode": "completion",
        "completion_authorized": False,
    }


@pytest.mark.asyncio
async def test_trivial_closeout_completes_without_decision_or_lessons(db_session, test_project):
    from huddleroom.services.orchestration_goal_closeout import GoalCloseoutProcess

    goal, run = await create_goal(db_session, test_project, weight="trivial")
    manifest = completion_manifest_fixture(run)

    result = await GoalCloseoutProcess().advance(db_session, goal, run, preconditions=manifest)

    assert result["status"] == "completed"
    assert result["completion_authorized"] is True
    assert result["full_closeout"] is False
    assert await pending_decisions(db_session, goal.id) == []
    assert await memory_section(db_session, goal, "completion_rationale") is not None
    assert await memory_section(db_session, goal, "lessons_learned") is None


@pytest.mark.asyncio
async def test_standard_closeout_parks_on_one_signoff_decision(db_session, test_project):
    from huddleroom.services.orchestration_goal_closeout import GoalCloseoutProcess

    goal, run = await create_goal(db_session, test_project, weight="standard")

    first = await GoalCloseoutProcess().advance(
        db_session, goal, run, preconditions=completion_manifest_fixture(run)
    )
    second = await GoalCloseoutProcess().advance(
        db_session, goal, run, preconditions=completion_manifest_fixture(run)
    )
    pending = await pending_decisions(db_session, goal.id)

    assert first["status"] == second["status"] == "waiting_decision"
    assert len(pending) == 1
    assert pending[0].decision_key.startswith("goal_closeout:signoff:")
    assert pending[0].options == [
        {"key": "approve_completion", "description": "Authorize run completion."},
        {"key": "keep_open", "description": "Leave the run active."},
    ]
    assert pending[0].recommendation == "approve_completion"


@pytest.mark.asyncio
async def test_closeout_routes_to_active_agent_manager(db_session, test_project, test_agent):
    from huddleroom.services.orchestration_goal_closeout import GoalCloseoutProcess

    goal, run = await create_goal(db_session, test_project, weight="substantial")
    goal.authority_model = "agent_manager"
    goal.manager_agent_id = test_agent.id

    await GoalCloseoutProcess().advance(
        db_session, goal, run, preconditions=completion_manifest_fixture(run)
    )
    decision = (await pending_decisions(db_session, goal.id))[0]

    assert decision.authority == "manager"
    assert decision.authority_agent_id == test_agent.id


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["no_manager", "human_manager", "deleted_manager", "inactive_manager"])
async def test_closeout_manager_fallbacks_route_to_human(
    db_session, test_project, test_agent, case
):
    from huddleroom.services.orchestration_goal_closeout import GoalCloseoutProcess

    goal, run = await create_goal(db_session, test_project, weight="standard")
    if case == "human_manager":
        goal.authority_model = "human_manager"
    elif case != "no_manager":
        goal.authority_model = "agent_manager"
        goal.manager_agent_id = test_agent.id
        if case == "deleted_manager":
            await db_session.flush()
            await db_session.delete(test_agent)
            await db_session.flush()
        else:
            test_agent.is_active = False

    await GoalCloseoutProcess().advance(
        db_session, goal, run, preconditions=completion_manifest_fixture(run)
    )
    decision = (await pending_decisions(db_session, goal.id))[0]

    assert decision.authority == "human"
    assert decision.authority_agent_id is None


@pytest.mark.asyncio
async def test_approved_signoff_writes_lessons_cleans_pending_and_authorizes(
    db_session, test_project, test_user
):
    from huddleroom.services.orchestration_goal_closeout import GoalCloseoutProcess

    goal, run = await create_goal(db_session, test_project, weight="standard")
    process = GoalCloseoutProcess()
    manifest = completion_manifest_fixture(run)
    await process.advance(db_session, goal, run, preconditions=manifest)
    signoff = (await pending_decisions(db_session, goal.id))[0]
    stale = await OrchestrationAuthorityDecisionService().create_pending(
        db_session,
        goal.id,
        decision_key="stale:question",
        title="Old question",
        question="Still relevant?",
        authority="human",
        run_id=run.id,
    )
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session,
        signoff,
        selected_option="approve_completion",
        reason="Evidence is sufficient.",
        decided_by_user_id=test_user.id,
    )

    result = await process.advance(db_session, goal, run, preconditions=manifest)

    assert result["status"] == "completed"
    assert result["completion_authorized"] is True
    assert result["selected_option"] == "approve_completion"
    await db_session.refresh(stale)
    assert stale.status == "cancelled"
    assert stale.reason == "goal closeout completed"
    assert await memory_section(db_session, goal, "completion_rationale") is not None
    lessons = await memory_section(db_session, goal, "lessons_learned")
    assert lessons is not None
    assert lessons.toc_order == 100


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_status", ["keep_open", "cancelled", "expired"])
async def test_non_approving_signoff_never_authorizes_completion(
    db_session, test_project, test_user, terminal_status
):
    from huddleroom.services.orchestration_goal_closeout import GoalCloseoutProcess

    goal, run = await create_goal(db_session, test_project, weight="standard")
    process = GoalCloseoutProcess()
    manifest = completion_manifest_fixture(run)
    await process.advance(db_session, goal, run, preconditions=manifest)
    decision = (await pending_decisions(db_session, goal.id))[0]
    if terminal_status == "keep_open":
        await OrchestrationAuthorityDecisionService().answer_decision(
            db_session,
            decision,
            selected_option="keep_open",
            reason="More work is required.",
            decided_by_user_id=test_user.id,
        )
    elif terminal_status == "cancelled":
        await OrchestrationAuthorityDecisionService().cancel_decision(
            db_session, decision, reason="Sign-off withdrawn."
        )
    else:
        decision.status = "expired"
        decision.reason = "Sign-off expired."
        await db_session.flush()

    result = await process.advance(db_session, goal, run, preconditions=manifest)
    terminal = await OrchestrationProcessService().get_current(
        db_session, goal.id, "goal_closeout"
    )
    repeated = await process.advance(db_session, goal, run, preconditions=manifest)

    assert result["status"] == "completed"
    assert result["completion_authorized"] is False
    assert repeated["status"] == "completed"
    assert repeated["completion_authorized"] is False
    assert (
        await OrchestrationProcessService().get_current(
            db_session, goal.id, "goal_closeout"
        )
    ).id == terminal.id
    assert await memory_section(db_session, goal, "completion_rationale") is None


@pytest.mark.asyncio
async def test_completion_authorization_requires_completed_authorized_closeout(
    db_session, test_project
):
    from huddleroom.services.orchestration_goal_closeout import GoalCloseoutProcess

    goal, run = await create_goal(db_session, test_project, weight="standard")
    process = GoalCloseoutProcess()

    with pytest.raises(HTTPException, match="409"):
        await process.completion_authorization(db_session, goal, run)


@pytest.mark.asyncio
async def test_completion_authorization_rejects_closeout_from_another_run(
    db_session, test_project
):
    from huddleroom.services.orchestration_goal_closeout import GoalCloseoutProcess

    goal, run = await create_goal(db_session, test_project, weight="trivial")
    _, other_run = await create_goal(db_session, test_project, weight="trivial")
    process = GoalCloseoutProcess()
    await process.advance(
        db_session, goal, run, preconditions=completion_manifest_fixture(run)
    )

    with pytest.raises(HTTPException) as exc_info:
        await process.completion_authorization(db_session, goal, other_run)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Goal closeout must authorize completion"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "gates",
    [None, {"closeout_completed": False}, "malformed"],
    ids=["missing", "false", "malformed"],
)
async def test_completion_authorization_requires_true_closeout_gate(
    db_session, test_project, gates
):
    from huddleroom.services.orchestration_goal_closeout import GoalCloseoutProcess

    goal, run = await create_goal(db_session, test_project, weight="trivial")
    process = GoalCloseoutProcess()
    await process.advance(
        db_session, goal, run, preconditions=completion_manifest_fixture(run)
    )
    current = await OrchestrationProcessService().get_current(
        db_session, goal.id, "goal_closeout"
    )
    outputs = dict(current.outputs)
    if gates is None:
        outputs.pop("gates")
    else:
        outputs["gates"] = gates
    current.outputs = outputs
    await db_session.flush()

    with pytest.raises(HTTPException) as exc_info:
        await process.completion_authorization(db_session, goal, run)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Goal closeout must authorize completion"


async def significant_action(db_session, run):
    action = OrchestrationAction(
        run_id=run.id,
        idempotency_key=f"cancel-closeout:{run.id}",
        action_type="expand_plan_item",
    )
    db_session.add(action)
    await db_session.flush()
    return action


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["active", "blocked", "paused"])
async def test_cancel_significant_goal_records_cancellation_closeout(
    db_session, test_project, test_user, state
):
    from huddleroom.services.orchestration_service import OrchestrationService

    goal, run = await create_goal(db_session, test_project, weight="standard")
    goal.status = state
    run.status = "running" if state == "active" else state
    await significant_action(db_session, run)
    pending = await OrchestrationAuthorityDecisionService().create_pending(
        db_session,
        goal.id,
        decision_key=f"cancel:stale:{state}",
        title="Stale question",
        question="Still needed?",
        authority="human",
        run_id=run.id,
    )

    returned_goal, returned_run = await OrchestrationService().cancel_goal(
        db_session,
        test_project.id,
        goal.id,
        cancelled_by=f"human:{test_user.id}",
    )

    assert returned_goal.status == "cancelled"
    assert returned_run.status == "cancelled"
    await db_session.refresh(pending)
    assert pending.status == "cancelled"
    rationale = await memory_section(db_session, goal, "completion_rationale")
    assert json.loads(rationale.body)["outcome"] == "cancelled"
    assert json.loads(rationale.body)["cancelled_by"] == f"human:{test_user.id}"
    assert json.loads(rationale.body)["significant_work"] is True
    closeout = await OrchestrationProcessService().get_current(db_session, goal.id, "goal_closeout")
    assert closeout.status == "completed"
    assert closeout.outputs["mode"] == "cancellation"
    assert await memory_section(db_session, goal, "lessons_learned") is None


@pytest.mark.asyncio
async def test_cancel_waiting_completion_closeout_completes_as_cancellation(
    db_session, test_project
):
    from huddleroom.services.orchestration_goal_closeout import GoalCloseoutProcess
    from huddleroom.services.orchestration_service import OrchestrationService

    goal, run = await create_goal(db_session, test_project, weight="standard")
    await significant_action(db_session, run)
    await GoalCloseoutProcess().advance(
        db_session, goal, run, preconditions=completion_manifest_fixture(run)
    )
    signoff = (await pending_decisions(db_session, goal.id))[0]

    await OrchestrationService().cancel_goal(db_session, test_project.id, goal.id)

    await db_session.refresh(signoff)
    assert signoff.status == "cancelled"
    closeout = await OrchestrationProcessService().get_current(db_session, goal.id, "goal_closeout")
    assert closeout.status == "completed"
    assert closeout.outputs["mode"] == "cancellation"
    assert closeout.outputs["completion_authorized"] is False
    # End this test-owned setup before the real orchestration operation.
    await db_session.flush()
    await db_session.commit()
    tick = await OrchestrationService().tick(db_session, run.id)
    assert tick["goal_closeout_process"]["mode"] == "cancellation"


@pytest.mark.asyncio
async def test_cancel_without_significant_work_skips_closeout_memory(db_session, test_project):
    from huddleroom.services.orchestration_service import OrchestrationService

    goal, run = await create_goal(db_session, test_project, weight="standard")
    pending = await OrchestrationAuthorityDecisionService().create_pending(
        db_session,
        goal.id,
        decision_key="cancel:insignificant",
        title="Unstarted question",
        question="Still needed?",
        authority="human",
        run_id=run.id,
    )

    await OrchestrationService().cancel_goal(db_session, test_project.id, goal.id)

    await db_session.refresh(pending)
    assert pending.status == "cancelled"
    assert await OrchestrationProcessService().get_current(db_session, goal.id, "goal_closeout") is None
    assert await memory_section(db_session, goal, "completion_rationale") is None
    assert await memory_section(db_session, goal, "lessons_learned") is None


@pytest.mark.asyncio
async def test_cancel_without_active_run_skips_closeout_memory(db_session, test_project):
    from huddleroom.services.orchestration_service import OrchestrationService

    goal, run = await create_goal(db_session, test_project, weight="standard")
    run.status = "completed"
    await db_session.flush()

    returned_goal, returned_run = await OrchestrationService().cancel_goal(
        db_session, test_project.id, goal.id
    )

    assert returned_goal.status == "cancelled"
    assert returned_run is None
    assert await OrchestrationProcessService().get_current(db_session, goal.id, "goal_closeout") is None
    assert await memory_section(db_session, goal, "completion_rationale") is None


@pytest.mark.asyncio
async def test_cancel_api_remains_bodyless_and_records_actor(
    client, db_session, test_project, test_user, auth_headers
):
    goal, run = await create_goal(db_session, test_project, weight="standard")
    await significant_action(db_session, run)

    response = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/cancel",
        headers=auth_headers,
    )

    assert response.status_code == 200
    rationale = await memory_section(db_session, goal, "completion_rationale")
    assert json.loads(rationale.body)["cancelled_by"] == f"human:{test_user.id}"


@pytest.mark.asyncio
async def test_cancel_rechecks_terminal_status_after_acquiring_goal_lock(
    db_session, test_project, monkeypatch
):
    from huddleroom.services.orchestration_service import OrchestrationService

    goal, run = await create_goal(db_session, test_project, weight="standard")
    service = OrchestrationService()

    @asynccontextmanager
    async def complete_before_locked_read(_db, _goal_id):
        goal.status = "completed"
        run.status = "completed"
        await db_session.flush()
        yield

    monkeypatch.setattr(service, "_lock_goal_for_baseline_transition", complete_before_locked_read)

    with pytest.raises(HTTPException) as exc:
        await service.cancel_goal(db_session, test_project.id, goal.id)

    assert exc.value.status_code == 409
    assert goal.status == "completed"
    assert run.status == "completed"


@pytest.mark.asyncio
async def test_cancel_waits_for_tick_goal_lock_and_cancels_pending_decision(
    concurrent_sessions, monkeypatch, tmp_path
):
    from huddleroom.models.project import Project
    from huddleroom.services.orchestration_service import OrchestrationService

    tick_db, cancel_db = concurrent_sessions
    project = Project(
        name=f"cancel-tick-lock-{uuid.uuid4()}", workspace_path=str(tmp_path), config={}
    )
    tick_db.add(project)
    await tick_db.flush()
    goal, run = await create_goal(tick_db, project, weight="standard")
    pending = await OrchestrationAuthorityDecisionService().create_pending(
        tick_db,
        goal.id,
        decision_key="cancel:during-tick",
        title="Pending question",
        question="Should this continue?",
        authority="human",
        run_id=run.id,
    )
    await tick_db.commit()

    await stub_tick_baseline(monkeypatch)
    service = OrchestrationService()
    sync = service._sync_agent_authority_decisions
    tick_entered = asyncio.Event()
    release_tick = asyncio.Event()
    cancel_attempted_lock = asyncio.Event()
    lock_goal = service._lock_goal_for_baseline_transition

    @asynccontextmanager
    async def observe_lock(db, goal_id, *, tick_owns_transaction=False):
        if db is cancel_db:
            cancel_attempted_lock.set()
        async with lock_goal(db, goal_id, tick_owns_transaction=tick_owns_transaction):
            yield

    async def pause_sync(*args, **kwargs):
        tick_entered.set()
        await release_tick.wait()
        return await sync(*args, **kwargs)

    monkeypatch.setattr(service, "_lock_goal_for_baseline_transition", observe_lock)
    monkeypatch.setattr(service, "_sync_agent_authority_decisions", pause_sync)
    tick_task = asyncio.create_task(service.tick(tick_db, run.id))
    await asyncio.wait_for(tick_entered.wait(), timeout=1)
    cancel_task = asyncio.create_task(service.cancel_goal(cancel_db, project.id, goal.id))
    await asyncio.wait_for(cancel_attempted_lock.wait(), timeout=1)
    assert not cancel_task.done()

    release_tick.set()
    await asyncio.wait_for(asyncio.gather(tick_task, cancel_task), timeout=5)

    cancelled_goal = await tick_db.get(OrchestrationGoal, goal.id, populate_existing=True)
    cancelled_run = await tick_db.get(OrchestrationRun, run.id, populate_existing=True)
    cancelled_decision = await tick_db.get(
        OrchestrationAuthorityDecision, pending.id, populate_existing=True
    )
    assert cancelled_goal.status == "cancelled"
    assert cancelled_run.status == "cancelled"
    assert cancelled_decision.status == "cancelled"


@pytest.mark.asyncio
async def test_repeated_cancel_returns_terminal_state_error(db_session, test_project):
    from huddleroom.services.orchestration_service import OrchestrationService

    goal, _run = await create_goal(db_session, test_project, weight="standard")
    service = OrchestrationService()

    await service.cancel_goal(db_session, test_project.id, goal.id)

    with pytest.raises(HTTPException) as exc:
        await service.cancel_goal(db_session, test_project.id, goal.id)

    assert exc.value.status_code == 409
    assert exc.value.detail == "Cannot cancel goal in status 'cancelled'"


@pytest.mark.asyncio
async def test_cancel_closeout_failure_leaves_goal_and_run_nonterminal(
    db_session, test_project, monkeypatch
):
    from huddleroom.services.orchestration_goal_closeout import GoalCloseoutProcess
    from huddleroom.services.orchestration_service import OrchestrationService

    goal, run = await create_goal(db_session, test_project, weight="standard")

    async def fail_closeout(*_args, **_kwargs):
        raise RuntimeError("closeout failed")

    monkeypatch.setattr(GoalCloseoutProcess, "close_cancelled", fail_closeout)

    with pytest.raises(RuntimeError, match="closeout failed"):
        await OrchestrationService().cancel_goal(db_session, test_project.id, goal.id)

    assert goal.status == "active"
    assert run.status == "running"


@pytest.mark.asyncio
async def test_closeout_blocks_on_unresolved_meeting_commitment(db_session, test_project, completion_ready_goal):
    from huddleroom.models.meeting import Meeting, MeetingActionItem
    from huddleroom.models.task import Task

    service, goal, run = completion_ready_goal
    source = Task(
        project_id=test_project.id, title="Source", status="done",
        metadata_={"orchestration": {"run_id": str(run.id)}},
    )
    failed = Task(
        project_id=test_project.id, title="Failed follow-up", status="failed",
        metadata_={"orchestration": {"run_id": str(run.id)}},
    )
    db_session.add_all([source, failed])
    await db_session.flush()
    meeting = Meeting(project_id=test_project.id, title="Sync", meeting_type="standup", source_task_id=source.id)
    db_session.add(meeting)
    await db_session.flush()
    item = MeetingActionItem(meeting_id=meeting.id, description="Do it", task_id=failed.id, status="task_created")
    db_session.add(item)
    await db_session.flush()

    with pytest.raises(HTTPException) as exc:
        await service._closeout_preconditions_manifest(db_session, goal, run)
    assert exc.value.status_code == 409
    assert "Unresolved meeting commitments" in exc.value.detail and str(item.id) in exc.value.detail

    item.status = "waived"
    await db_session.flush()
    manifest = await service._closeout_preconditions_manifest(db_session, goal, run)
    assert manifest["unresolved_meeting_commitments"] == []


@pytest.mark.asyncio
async def test_tick_swallows_commitment_409_and_progress_view_lists_it(db_session, test_project, completion_ready_goal):
    from huddleroom.models.meeting import Meeting, MeetingActionItem
    from huddleroom.models.task import Task
    from huddleroom.services.orchestration_progress_view import OrchestrationProgressView

    service, goal, run = completion_ready_goal
    source = Task(
        project_id=test_project.id, title="Source", status="done",
        metadata_={"orchestration": {"run_id": str(run.id)}},
    )
    db_session.add(source)
    await db_session.flush()
    meeting = Meeting(project_id=test_project.id, title="Sync", meeting_type="standup", source_task_id=source.id)
    db_session.add(meeting)
    await db_session.flush()
    item = MeetingActionItem(meeting_id=meeting.id, description="Unhandled commitment")
    db_session.add(item)
    await db_session.flush()

    # End this test-owned setup before the real orchestration operation.
    await db_session.flush()
    await db_session.commit()
    await service.tick(db_session, run.id)

    assert run.status != "completed"
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    assert str(item.id) in {f["id"] for f in situation.untracked_follow_ups}


@pytest.mark.asyncio
async def test_preliminary_evidence_never_makes_closeout_ready_until_final_verification(db_session, test_project):
    from huddleroom.models.orchestration import OrchestrationEvidence, OrchestrationGate
    from huddleroom.services.orchestration_service import OrchestrationService

    service = OrchestrationService()
    goal, run = await create_goal(db_session, test_project, weight="standard")

    async def accepted_gate(extra):
        gate = OrchestrationGate(
            run_id=run.id, success_criterion_key="plan_item:done", gate_type="work_completed",
            required_evidence={"success_criterion_keys": ["done"], **extra}, status="accepted",
        )
        db_session.add(gate)
        await db_session.flush()
        db_session.add(OrchestrationEvidence(
            run_id=run.id, gate_id=gate.id, source_type="verification", source_id=goal.id,
            verdict="accepted", evidence_metadata={},
        ))
        await db_session.flush()

    await accepted_gate({"review_stage": "preliminary"})
    assert await service._run_ready_for_final_summary_request(db_session, goal, run) is False

    await accepted_gate({})
    assert await service._run_ready_for_final_summary_request(db_session, goal, run) is True


@pytest.mark.asyncio
async def test_closeout_decision_context_includes_only_applied_steering(
    db_session, test_project, test_agent, test_user, monkeypatch
):
    from huddleroom.config import settings
    from huddleroom.models.orchestration_steering import OrchestrationSteeringRequest
    from huddleroom.services.orchestration_goal_closeout import GoalCloseoutProcess

    monkeypatch.setattr(settings, "orchestration_conversation_steering_enabled", True)
    goal, run = await create_goal(db_session, test_project, weight="substantial")
    goal.authority_model = "agent_manager"
    goal.manager_agent_id = test_agent.id
    ids = {}
    for sequence, status in enumerate(
        ["applied", "superseded", "withdrawn", "pending", "needs_clarification"], start=1
    ):
        request = OrchestrationSteeringRequest(
            id=uuid.uuid4(), goal_id=goal.id, actor_id=test_user.id, client_request_id=uuid.uuid4(),
            sequence=sequence, submitted_run_id=run.id, directive=f"directive-{status}",
            target_type="goal", target_id=str(goal.id), scope="goal", lifetime="future_runs",
            impact_summary="x", status=status, reason_code="submitted",
            contract_version="none", plan_version="none",
        )
        db_session.add(request)
        ids[status] = request.id
    await db_session.flush()

    await GoalCloseoutProcess().advance(
        db_session, goal, run, preconditions=completion_manifest_fixture(run)
    )
    decision = (await pending_decisions(db_session, goal.id))[0]
    steering = json.loads(decision.context)["steering"]

    assert [d["directive"] for d in steering["active_directions"]] == ["directive-applied"]
    assert steering["active_directions"][0]["request_id"] == str(ids["applied"])
    for status in ("superseded", "withdrawn", "pending", "needs_clarification"):
        assert f"directive-{status}" not in decision.context


# --- SPR 285: persisted closeout diagnostic -------------------------------------------------------
async def _active_warnings(db, goal_id, warning_type=None):
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    rows = await OrchestrationWarningService().list_warnings(db, goal_id, active_only=True)
    return [w for w in rows if warning_type is None or w.warning_type == warning_type]


async def _all_warnings(db, goal_id, warning_type):
    from huddleroom.models.orchestration_process import OrchestrationWarning

    return list((await db.execute(
        select(OrchestrationWarning).where(
            OrchestrationWarning.goal_id == goal_id, OrchestrationWarning.warning_type == warning_type
        ).order_by(OrchestrationWarning.created_at, OrchestrationWarning.id)
    )).scalars().all())


async def _add_warning(db, goal, run, warning_type="test_blocker", severity="blocker", mirror=False):
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    return await OrchestrationWarningService().create_warning(
        db, goal.id, warning_type=warning_type, severity=severity,
        message=f"{warning_type} problem", run_id=run.id,
    )


async def _tick(db, service, run):
    await db.flush()
    await db.commit()
    result = await service.tick(db, run.id)
    await db.refresh(run)
    return result


@pytest.fixture
async def blocked_closeout(db_session, completion_ready_goal, monkeypatch):
    from huddleroom.models.artifact import Artifact
    from huddleroom.schemas.orchestration import OrchestrationPlanItem

    adopt_session(db_session)
    service, goal, run = completion_ready_goal
    artifact = Artifact(
        project_id=goal.project_id, name="accepted-plan", artifact_type="plan",
        metadata_={"kind": "implementation_plan"},
    )
    db_session.add(artifact)
    await db_session.flush()
    item = OrchestrationPlanItem(id="done", work_function="implementation", scope="Do it", deliverable="Done", success_criterion_keys=["done"])
    from huddleroom.models.orchestration import OrchestrationGate
    from huddleroom.models.task import Task

    work = Task(project_id=goal.project_id, title="Do it", status="done", metadata_={})
    db_session.add(work)
    await db_session.flush()
    gate_id = await db_session.scalar(
        select(OrchestrationGate.id).where(
            OrchestrationGate.run_id == run.id, OrchestrationGate.success_criterion_key == "plan_item:done"
        )
    )
    run.plan_state = {
        "expanded_items": [{"plan_item_id": "done", "task_id": str(work.id), "gate_id": str(gate_id)}],
        "status": "accepted",
        "accepted_artifact_id": str(artifact.id),
        "accepted_plan_snapshot": service._accepted_plan_snapshot(artifact.id, [item]),
    }
    await stub_tick_baseline(monkeypatch)

    async def no_release(*_args):
        return 0

    async def bound(*_args):
        return True

    monkeypatch.setattr(service, "_release_ready_work", no_release)
    # The fixture's accepted work gate has no verifier session; its binding is not under test here.
    monkeypatch.setattr(service, "_has_bound_outcome_verification_evidence", bound)
    return service, goal, run


@pytest.fixture
def no_provider(monkeypatch):
    """Any provider call during a tick fails the test; the diagnostic must be deterministic."""
    import litellm

    calls = []

    async def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("provider call")

    monkeypatch.setattr(litellm, "acompletion", forbidden)
    return calls


def _human():
    return f"human:{uuid.uuid4()}"


async def _resolve(db, warning, by=None, reason="fixed"):
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    await OrchestrationWarningService().resolve_warning(db, warning, resolved_by=by or _human(), reason=reason)
    await db.flush()


def _assert_types_only(message, *types):
    import re

    for warning_type in types:
        assert warning_type in message
    assert not re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", message)


def _kinds(run):
    return [b.get("kind") for b in run.active_blockers or [] if isinstance(b, dict)]


@pytest.mark.asyncio
async def test_blocker_warning_gets_one_actionable_closeout_diagnostic(db_session, blocked_closeout, no_provider):
    service, goal, run = blocked_closeout
    source = await _add_warning(db_session, goal, run, "test_blocker")

    result = await _tick(db_session, service, run)

    assert result["authorized_execution"]["outcome"] == "closeout_ready"
    assert result["run_completed"] is False and result["goal_closeout_process"] is None
    (diag,) = await _active_warnings(db_session, goal.id, "closeout_blocked")
    assert diag.severity == "blocker" and diag.run_id == run.id
    _assert_types_only(diag.message, "test_blocker")
    assert diag.message.startswith("Goal closeout cannot start until these are resolved:")
    assert f"warning:{diag.id}" not in _kinds(run)
    assert f"warning:{source.id}" in _kinds(run)
    entry = run.supervision_state["closeout_attention"]
    assert entry["warning_id"] == str(diag.id)
    assert entry["source_warning_ids"] == [str(source.id)]  # ids live in state, never in the copy
    assert goal.status == "active" and run.status == "running"
    assert no_provider == []


@pytest.mark.asyncio
async def test_repeated_ticks_keep_one_unchanged_diagnostic(db_session, blocked_closeout, no_provider):
    service, goal, run = blocked_closeout
    await _add_warning(db_session, goal, run)
    await _tick(db_session, service, run)
    (first,) = await _active_warnings(db_session, goal.id, "closeout_blocked")
    first_id = first.id

    for _ in range(4):
        await _tick(db_session, service, run)

    (again,) = await _active_warnings(db_session, goal.id, "closeout_blocked")
    assert again.id == first_id
    assert len(await _all_warnings(db_session, goal.id, "closeout_blocked")) == 1
    assert _kinds(run).count(f"warning:{first_id}") == 0
    assert no_provider == []


@pytest.mark.asyncio
async def test_concurrent_ticks_create_a_single_diagnostic(
    db_session, blocked_closeout, owned_session_factory, no_provider
):
    from tests.conftest import owned_session

    service, goal, run = blocked_closeout
    await _add_warning(db_session, goal, run)
    await db_session.flush()
    await db_session.commit()

    async def tick_once():
        async with owned_session(owned_session_factory) as db:
            return await service.tick(db, run.id)

    await asyncio.gather(tick_once(), tick_once())

    assert len(await _all_warnings(db_session, goal.id, "closeout_blocked")) == 1
    assert len(await _active_warnings(db_session, goal.id, "closeout_blocked")) == 1


@pytest.mark.asyncio
async def test_diagnostic_never_blocks_its_own_preflight(db_session, blocked_closeout):
    service, goal, run = blocked_closeout
    source = await _add_warning(db_session, goal, run)
    await _tick(db_session, service, run)
    assert len(await _active_warnings(db_session, goal.id, "closeout_blocked")) == 1
    with pytest.raises(HTTPException, match="test_blocker"):
        await service._closeout_preconditions_manifest(db_session, goal, run)

    await _resolve(db_session, source)

    manifest = await service._closeout_preconditions_manifest(db_session, goal, run)  # diagnostic still active
    assert all(item["type"] != "closeout_blocked" for item in manifest["warning_disposition"])
    assert len(await _active_warnings(db_session, goal.id, "closeout_blocked")) == 1


@pytest.mark.asyncio
async def test_manually_resolving_only_the_diagnostic_restores_it_next_tick(db_session, blocked_closeout):
    service, goal, run = blocked_closeout
    source = await _add_warning(db_session, goal, run)
    await _tick(db_session, service, run)
    (diag,) = await _active_warnings(db_session, goal.id, "closeout_blocked")

    await _resolve(db_session, diag)

    await db_session.refresh(source)
    assert source.active is True
    assert await service._run_ready_for_completion(db_session, goal, run) is False
    result = await _tick(db_session, service, run)
    assert result["run_completed"] is False and goal.status == "active"
    (restored,) = await _active_warnings(db_session, goal.id, "closeout_blocked")
    assert restored.id != diag.id
    assert run.supervision_state["closeout_attention"]["warning_id"] == str(restored.id)
    assert f"warning:{restored.id}" not in _kinds(run)


@pytest.mark.asyncio
async def test_resolving_the_cause_retires_diagnostic_and_closeout_still_needs_authority(
    db_session, blocked_closeout, test_user
):
    service, goal, run = blocked_closeout
    source = await _add_warning(db_session, goal, run)
    await _tick(db_session, service, run)
    (diag,) = await _active_warnings(db_session, goal.id, "closeout_blocked")

    await _resolve(db_session, source)
    result = await _tick(db_session, service, run)

    await db_session.refresh(diag)
    assert diag.active is False and diag.resolved_reason == "precondition cleared"
    assert "closeout_attention" not in run.supervision_state
    assert f"warning:{diag.id}" not in _kinds(run)
    assert result["goal_closeout_process"]["status"] == "waiting_decision"  # closeout advances ...
    assert result["run_completed"] is False and goal.status == "active"  # ... but is not self-authorized
    decision = await db_session.get(
        OrchestrationAuthorityDecision, uuid.UUID(result["goal_closeout_process"]["decision_id"])
    )
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, decision, selected_option="approve_completion", reason="ok", decided_by_user_id=test_user.id
    )
    done = await _tick(db_session, service, run)
    assert done["run_completed"] is True
    assert await _active_warnings(db_session, goal.id, "closeout_blocked") == []


@pytest.mark.asyncio
async def test_changed_blocker_set_replaces_the_diagnostic(db_session, blocked_closeout):
    service, goal, run = blocked_closeout
    first = await _add_warning(db_session, goal, run, "first_blocker")
    await _tick(db_session, service, run)
    (old,) = await _active_warnings(db_session, goal.id, "closeout_blocked")
    second = await _add_warning(db_session, goal, run, "second_blocker")

    await _tick(db_session, service, run)

    await db_session.refresh(old)
    assert old.active is False and old.resolved_reason == "conditions changed"
    (new,) = await _active_warnings(db_session, goal.id, "closeout_blocked")
    assert new.id != old.id
    _assert_types_only(new.message, "first_blocker", "second_blocker")
    assert run.supervision_state["closeout_attention"]["source_warning_ids"] == sorted([str(first.id), str(second.id)])

    await _resolve(db_session, first)  # shrinking the set is also a change
    await _tick(db_session, service, run)
    await db_session.refresh(new)
    assert new.active is False and new.resolved_reason == "conditions changed"
    (newest,) = await _active_warnings(db_session, goal.id, "closeout_blocked")
    _assert_types_only(newest.message, "second_blocker")
    assert "first_blocker" not in newest.message
    assert run.supervision_state["closeout_attention"]["source_warning_ids"] == [str(second.id)]
    assert len(await _all_warnings(db_session, goal.id, "closeout_blocked")) == 3


@pytest.mark.asyncio
async def test_cleared_cause_that_recurs_gets_a_fresh_diagnostic(db_session, blocked_closeout):
    # Driven through the sync itself: a real clearing tick starts closeout, whose pending decision then
    # (correctly) parks liveness in a wait instead of closeout_ready.
    service, goal, run = blocked_closeout
    source = await _add_warning(db_session, goal, run)
    first = await service._sync_closeout_attention(db_session, goal, run, eligible=True)
    assert first is not None and first.active
    await _resolve(db_session, source)

    assert await service._sync_closeout_attention(db_session, goal, run, eligible=True) is None

    await db_session.refresh(first)
    assert first.active is False and first.resolved_reason == "precondition cleared"
    assert "closeout_attention" not in run.supervision_state
    again = await _add_warning(db_session, goal, run)
    second = await service._sync_closeout_attention(db_session, goal, run, eligible=True)
    assert second is not None and second.id != first.id
    _assert_types_only(second.message, "test_blocker")
    assert run.supervision_state["closeout_attention"]["source_warning_ids"] == [str(again.id)]
    assert f"warning:{second.id}" not in _kinds(run)
    assert len(await _active_warnings(db_session, goal.id, "closeout_blocked")) == 1


@pytest.mark.asyncio
async def test_unacknowledged_advisory_warning_says_acknowledge_or_resolve(db_session, blocked_closeout):
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    service, goal, run = blocked_closeout
    advisory = await _add_warning(db_session, goal, run, "advisory_note", severity="warning")

    await _tick(db_session, service, run)

    (diag,) = await _active_warnings(db_session, goal.id, "closeout_blocked")
    assert "Acknowledge or Resolve" in diag.message and "advisory_note" in diag.message
    assert diag.severity == "blocker"  # acknowledging the diagnostic must not masquerade as the fix
    await OrchestrationWarningService().acknowledge_warning(db_session, advisory, acknowledged_by=_human())
    await db_session.flush()
    await _tick(db_session, service, run)
    await db_session.refresh(diag)
    assert diag.active is False and diag.resolved_reason == "precondition cleared"


@pytest.mark.asyncio
async def test_non_warning_conflict_is_diagnosed_only_on_second_eligible_tick(db_session, blocked_closeout):
    from huddleroom.models.orchestration import OrchestrationEvidence

    service, goal, run = blocked_closeout
    rows = (await db_session.execute(
        select(OrchestrationEvidence).where(OrchestrationEvidence.source_type == "verification")
    )).scalars().all()
    for row in rows:
        await db_session.delete(row)  # accepted work gate loses its evidence: a pure non-warning 409
    await db_session.flush()
    with pytest.raises(HTTPException, match="Accepted gate evidence is missing"):
        await service._closeout_preconditions_manifest(db_session, goal, run)

    await _tick(db_session, service, run)
    assert await _active_warnings(db_session, goal.id, "closeout_blocked") == []
    await _tick(db_session, service, run)

    (diag,) = await _active_warnings(db_session, goal.id, "closeout_blocked")
    assert "accepted evidence must be attached to every gate" in diag.message
    assert run.supervision_state["closeout_attention"]["source_warning_ids"] == []
    await _tick(db_session, service, run)
    await _tick(db_session, service, run)
    (same,) = await _active_warnings(db_session, goal.id, "closeout_blocked")
    assert same.id == diag.id
    assert len(await _all_warnings(db_session, goal.id, "closeout_blocked")) == 1


@pytest.mark.asyncio
async def test_non_conflict_failure_propagates_and_creates_nothing(db_session, blocked_closeout, monkeypatch):
    service, goal, run = blocked_closeout

    async def boom(*_args):
        raise HTTPException(status_code=500, detail="storage down")

    monkeypatch.setattr(service, "_closeout_preconditions_manifest", boom)

    with pytest.raises(HTTPException) as exc:
        await service._sync_closeout_attention(db_session, goal, run, eligible=True)

    assert exc.value.status_code == 500
    assert await _all_warnings(db_session, goal.id, "closeout_blocked") == []


@pytest.mark.asyncio
async def test_paused_run_retires_the_diagnostic_and_resume_restores_it(db_session, blocked_closeout):
    service, goal, run = blocked_closeout
    await _add_warning(db_session, goal, run)
    await _tick(db_session, service, run)
    (diag,) = await _active_warnings(db_session, goal.id, "closeout_blocked")

    run.status = "paused"
    await _tick(db_session, service, run)

    await db_session.refresh(diag)
    assert diag.active is False and diag.resolved_reason == "closeout no longer pending"
    assert "closeout_attention" not in (run.supervision_state or {})

    run.status = "running"
    await _tick(db_session, service, run)
    (restored,) = await _active_warnings(db_session, goal.id, "closeout_blocked")
    assert restored.id != diag.id


@pytest.mark.asyncio
async def test_ineligible_sync_retires_obsolete_diagnostic(db_session, blocked_closeout):
    service, goal, run = blocked_closeout
    await _add_warning(db_session, goal, run)
    await _tick(db_session, service, run)
    (diag,) = await _active_warnings(db_session, goal.id, "closeout_blocked")

    assert await service._sync_closeout_attention(db_session, goal, run, eligible=False) is None

    await db_session.refresh(diag)
    assert diag.active is False and diag.resolved_reason == "closeout no longer pending"
    assert "closeout_attention" not in run.supervision_state


@pytest.mark.asyncio
async def test_blocker_in_another_goal_creates_no_diagnostic_here(db_session, blocked_closeout, test_project):
    service, goal, run = blocked_closeout
    other_goal, other_run = await create_goal(db_session, test_project, weight="standard")
    other_run.status, other_run.phase = "running", "authorized"
    await _add_warning(db_session, other_goal, other_run, "other_goal_blocker")

    result = await _tick(db_session, service, run)

    assert await _all_warnings(db_session, goal.id, "closeout_blocked") == []
    assert await _all_warnings(db_session, other_goal.id, "closeout_blocked") == []
    assert result["goal_closeout_process"]["status"] == "waiting_decision"


@pytest.mark.asyncio
async def test_model_cannot_record_the_reserved_warning_type(db_session, blocked_closeout):
    service, goal, run = blocked_closeout

    with pytest.raises(HTTPException) as exc:
        await service.execute_record_warning_action(
            db_session,
            run.id,
            {"action_type": "record_warning", "warning_type": "closeout_blocked", "severity": "warning", "message": "x"},
            f"run:{run.id}:test:reserved",
        )

    assert exc.value.status_code == 409 and "reserved" in exc.value.detail
    assert await _all_warnings(db_session, goal.id, "closeout_blocked") == []


@pytest.mark.asyncio
async def test_normal_completion_never_creates_a_closeout_diagnostic(db_session, blocked_closeout, test_user):
    service, goal, run = blocked_closeout

    first = await _tick(db_session, service, run)
    decision = await db_session.get(
        OrchestrationAuthorityDecision, uuid.UUID(first["goal_closeout_process"]["decision_id"])
    )
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, decision, selected_option="approve_completion", reason="ok", decided_by_user_id=test_user.id
    )
    second = await _tick(db_session, service, run)

    assert second["run_completed"] is True and goal.status == "completed"
    assert await _all_warnings(db_session, goal.id, "closeout_blocked") == []
    assert "closeout_attention" not in (run.supervision_state or {})


@pytest.mark.asyncio
@pytest.mark.parametrize("cause", ["hard_stop", "budget_exhausted"])
async def test_early_return_retires_diagnostic_and_leaves_source_untouched(
    db_session, blocked_closeout, no_provider, cause
):
    service, goal, run = blocked_closeout
    source = await _add_warning(db_session, goal, run, "test_blocker")
    await _tick(db_session, service, run)
    (diag,) = await _active_warnings(db_session, goal.id, "closeout_blocked")

    if cause == "hard_stop":
        await _add_warning(db_session, goal, run, "emergency", severity="hard_stop")
    else:
        run.budget_state = {"status": "exceeded"}
    await _tick(db_session, service, run)

    await db_session.refresh(diag)
    await db_session.refresh(source)
    assert diag.active is False and diag.resolved_reason == "closeout no longer pending"
    assert "closeout_attention" not in (run.supervision_state or {})
    assert await _active_warnings(db_session, goal.id, "closeout_blocked") == []
    assert source.active is True
    assert f"warning:{source.id}" in _kinds(run) and f"warning:{diag.id}" not in _kinds(run)
    assert no_provider == []


@pytest.mark.asyncio
async def test_hard_stop_warning_is_in_the_blocking_set(db_session, blocked_closeout):
    service, goal, run = blocked_closeout
    hard = await _add_warning(db_session, goal, run, "emergency", severity="hard_stop")

    diag = await service._sync_closeout_attention(db_session, goal, run, eligible=True)

    assert diag is not None
    _assert_types_only(diag.message, "emergency")
    assert "Resolve: emergency" in diag.message
    assert run.supervision_state["closeout_attention"]["source_warning_ids"] == [str(hard.id)]


@pytest.mark.asyncio
async def test_warnings_and_unreleased_plan_items_are_both_named(db_session, blocked_closeout, monkeypatch):
    from huddleroom.services import orchestration_progress_view

    service, goal, run = blocked_closeout
    await _add_warning(db_session, goal, run, "test_blocker")
    monkeypatch.setattr(orchestration_progress_view, "unreleased_plan_item_ids", lambda _run: ["item-a", "item-b"])

    diag = await service._sync_closeout_attention(db_session, goal, run, eligible=True)

    assert "test_blocker" in diag.message
    assert "Accepted plan items are unreleased: item-a, item-b" in diag.message


@pytest.mark.asyncio
async def test_failed_gate_without_final_gate_is_named_not_blamed_on_evidence(db_session, blocked_closeout):
    from huddleroom.models.orchestration import OrchestrationEvidence, OrchestrationGate

    service, goal, run = blocked_closeout
    gates = list((await db_session.execute(
        select(OrchestrationGate).where(OrchestrationGate.run_id == run.id)
    )).scalars())
    for gate in gates:
        if gate.gate_type == "final_summary":
            for ev in (await db_session.execute(
                select(OrchestrationEvidence).where(OrchestrationEvidence.gate_id == gate.id)
            )).scalars():
                await db_session.delete(ev)
            await db_session.flush()
            await db_session.delete(gate)
        else:
            gate.status = "failed"
    await db_session.flush()
    with pytest.raises(HTTPException, match="All orchestration gates must be accepted"):
        await service._closeout_preconditions_manifest(db_session, goal, run)

    assert await service._sync_closeout_attention(db_session, goal, run, eligible=True) is None  # streak tick 1
    diag = await service._sync_closeout_attention(db_session, goal, run, eligible=True)

    assert "a gate failed:" in diag.message and "work_completed" in diag.message
    assert "lacks evidence" not in diag.message and "final summary could not be requested" not in diag.message


@pytest.mark.asyncio
async def test_replacing_failed_gate_changes_fingerprint_and_rediagnoses(db_session, blocked_closeout):
    from huddleroom.models.orchestration import OrchestrationEvidence, OrchestrationGate

    service, goal, run = blocked_closeout
    gates = list((await db_session.execute(
        select(OrchestrationGate).where(OrchestrationGate.run_id == run.id)
    )).scalars())
    failed_a = []
    for gate in gates:
        if gate.gate_type == "final_summary":
            for ev in (await db_session.execute(
                select(OrchestrationEvidence).where(OrchestrationEvidence.gate_id == gate.id)
            )).scalars():
                await db_session.delete(ev)
            await db_session.flush()
            await db_session.delete(gate)
        else:
            gate.status = "failed"
            failed_a.append(gate)
    await db_session.flush()

    assert await service._sync_closeout_attention(db_session, goal, run, eligible=True) is None  # streak tick 1
    old = await service._sync_closeout_attention(db_session, goal, run, eligible=True)
    old_id = old.id
    assert "work_completed" in old.message
    old_fp = run.supervision_state["closeout_attention"]["fingerprint"]

    for gate in failed_a:
        gate.status = "accepted"  # A is no longer failed
    db_session.add(OrchestrationGate(
        run_id=run.id, success_criterion_key="plan_item:other", gate_type="other_check",
        required_evidence={"success_criterion_keys": ["other"]}, status="failed",
    ))
    await db_session.flush()

    await service._sync_closeout_attention(db_session, goal, run, eligible=True)  # streak tick 1 for new state
    new = await service._sync_closeout_attention(db_session, goal, run, eligible=True)

    assert run.supervision_state["closeout_attention"]["fingerprint"] != old_fp
    assert new.id != old_id
    assert "other_check" in new.message and "work_completed" not in new.message
    (active,) = await _active_warnings(db_session, goal.id, "closeout_blocked")
    assert active.id == new.id
    old_row = next(w for w in await _all_warnings(db_session, goal.id, "closeout_blocked") if w.id == old_id)
    assert not old_row.active and "conditions changed" in (old_row.resolved_reason or "")


@pytest.mark.asyncio
async def test_closeout_attention_state_does_not_change_decision_context_fingerprint(blocked_closeout):
    service, _goal, run = blocked_closeout
    base = {"mode": "x", "last_pass_at": "t1"}
    with_entry = {**base, "closeout_attention": {"fingerprint": "f", "warning_id": "w", "source_warning_ids": ["a"]}}
    assert service._stable_supervision_state(with_entry) == service._stable_supervision_state(base) == {"mode": "x"}


@pytest.mark.asyncio
async def test_replayed_reserved_warning_type_is_rejected_and_action_failed(db_session, blocked_closeout):
    service, goal, run = blocked_closeout
    key = f"run:{run.id}:test:replay-reserved"
    await service.reserve_action(
        db_session, run.id, key, "record_warning",
        {"warning_type": "closeout_blocked", "severity": "blocker", "message": "forged"},
    )

    with pytest.raises(HTTPException) as exc:
        await service.execute_record_warning_action(
            db_session, run.id,
            {"action_type": "record_warning", "warning_type": "other", "severity": "warning", "message": "y"}, key,
        )

    assert exc.value.status_code == 409 and "reserved" in exc.value.detail
    assert await _all_warnings(db_session, goal.id, "closeout_blocked") == []


@pytest.mark.asyncio
async def test_source_mirror_survives_diagnostic_retirement_and_pause(db_session, blocked_closeout):
    service, goal, run = blocked_closeout
    source = await _add_warning(db_session, goal, run)
    await _tick(db_session, service, run)
    assert f"warning:{source.id}" in _kinds(run)

    run.status = "paused"
    await _tick(db_session, service, run)
    assert await _active_warnings(db_session, goal.id, "closeout_blocked") == []
    assert source.active is True and f"warning:{source.id}" in _kinds(run)

    run.status = "running"
    await _tick(db_session, service, run)
    assert f"warning:{source.id}" in _kinds(run)
    assert len(await _active_warnings(db_session, goal.id, "closeout_blocked")) == 1


# --- SPR 284 + 285: proven-cleared recovery warnings leave before closeout attention is evaluated -----------------


async def _recovery_lineage(db, goal, run, *, task_status):
    from huddleroom.models.task import Task
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    task = Task(project_id=goal.project_id, title="Retried work", status=task_status,
                metadata_={"orchestration": {"run_id": str(run.id)}})
    db.add(task)
    await db.flush()
    source = OrchestrationAction(
        run_id=run.id, idempotency_key=f"seed:{uuid.uuid4()}:failure", action_type="failure_observed", status="failed",
        request={"task_id": str(task.id), "session_id": str(uuid.uuid4()), "closes_when_task_healthy": True},
        error="boom",
        target_type="task", target_id=task.id,
    )
    db.add(source)
    await db.flush()
    root = OrchestrationAction(
        run_id=run.id, idempotency_key=f"seed:{uuid.uuid4()}:trigger", action_type="recovery_trigger", status="completed",
        request={
            "schema_version": 1, "goal_id": str(goal.id), "run_id": str(run.id), "source_kind": "failed_action",
            "source_id": str(source.id), "blocker_code": "task_failed", "summary": "The task failed.",
            "source_fingerprint": uuid.uuid4().hex + uuid.uuid4().hex, "retry_target": {"kind": "decision_loop"},
        },
    )
    db.add(root)
    await db.flush()
    return await OrchestrationWarningService().create_warning(
        db, goal.id, warning_type="recovery_blocked", severity="blocker", message="Recovery is blocked.",
        run_id=run.id, related_action_id=root.id,
    )


@pytest.mark.asyncio
async def test_cleared_recovery_warning_leaves_before_closeout_diagnostic_and_ambiguous_one_remains(
    db_session, blocked_closeout, no_provider,
):
    service, goal, run = blocked_closeout
    cleared = await _recovery_lineage(db_session, goal, run, task_status="done")
    ambiguous = await _add_warning(db_session, goal, run, "summary_gate_reconciliation_unavailable")

    await _tick(db_session, service, run)

    await db_session.refresh(cleared)
    assert not cleared.active and cleared.resolved_by == "orchestrator:recovery"
    assert f"warning:{cleared.id}" not in _kinds(run)
    (diag,) = await _active_warnings(db_session, goal.id, "closeout_blocked")
    entry = run.supervision_state["closeout_attention"]
    assert entry["source_warning_ids"] == [str(ambiguous.id)]  # the cleared one is never a listed cause
    _assert_types_only(diag.message, "summary_gate_reconciliation_unavailable")
    assert "recovery_blocked" not in diag.message

    await _resolve(db_session, ambiguous)  # supported manual resolution unlocks ordinary closeout
    await _tick(db_session, service, run)
    assert await _active_warnings(db_session, goal.id, "closeout_blocked") == []
    assert f"warning:{ambiguous.id}" not in _kinds(run)
    assert no_provider == []


@pytest.mark.asyncio
async def test_diagnostic_lists_only_the_recovery_warning_that_is_not_proven_cleared(db_session, blocked_closeout, no_provider):
    service, goal, run = blocked_closeout
    cleared = await _recovery_lineage(db_session, goal, run, task_status="done")
    still = await _recovery_lineage(db_session, goal, run, task_status="cancelled")

    await _tick(db_session, service, run)

    await db_session.refresh(cleared)
    await db_session.refresh(still)
    assert not cleared.active and still.active
    (diag,) = await _active_warnings(db_session, goal.id, "closeout_blocked")
    assert run.supervision_state["closeout_attention"]["source_warning_ids"] == [str(still.id)]
    assert diag.run_id == run.id


@pytest.mark.asyncio
@pytest.mark.parametrize("points_at", ["older", "newer"])
async def test_sync_keeps_the_state_pointed_diagnostic_and_rewrites_warning_id(
    db_session, blocked_closeout, no_provider, points_at,
):
    """Two active diagnostics (dedupe/lost update): exactly the one named by state survives, state agrees, and the
    presenter links it to its source warning."""
    from huddleroom.services.orchestration_needs_you_presenter import attach_warning_references
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    service, goal, run = blocked_closeout
    gate = (await service._current_run_gates(db_session, run.id))[0]
    source = await OrchestrationWarningService().create_warning(
        db_session, goal.id, warning_type="test_blocker", severity="blocker", message="src", run_id=run.id,
        related_gate_id=gate.id,
    )
    await _tick(db_session, service, run)
    (older,) = await _active_warnings(db_session, goal.id, "closeout_blocked")
    from huddleroom.models.orchestration_process import OrchestrationWarning

    newer = OrchestrationWarning(  # bypasses create_warning's dedupe, as a lost update would
        goal_id=goal.id, run_id=run.id, warning_type="closeout_blocked", severity="blocker",
        message="duplicate diagnostic", active=True,
    )
    db_session.add(newer)
    await db_session.flush()
    service._remove_active_blocker_by_kind(run, f"warning:{newer.id}")
    kept = older if points_at == "older" else newer
    state = dict(run.supervision_state)
    state["closeout_attention"] = {**state["closeout_attention"], "warning_id": str(kept.id)}
    run.supervision_state = state
    await db_session.flush()
    assert len(await _active_warnings(db_session, goal.id, "closeout_blocked")) == 2

    returned = await service._sync_closeout_attention(db_session, goal, run, eligible=True)

    (only,) = await _active_warnings(db_session, goal.id, "closeout_blocked")
    assert only.id == kept.id == returned.id
    assert run.supervision_state["closeout_attention"]["warning_id"] == str(kept.id)
    assert run.supervision_state["closeout_attention"]["source_warning_ids"] == [str(source.id)]
    [response] = await attach_warning_references(db_session, project_id=goal.project_id, warnings=[only])
    assert ("gate", gate.id) in {(r.kind, r.id) for r in response.needs_you_references}
