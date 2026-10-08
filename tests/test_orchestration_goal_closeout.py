import asyncio
import json
import uuid
from contextlib import asynccontextmanager

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGoal, OrchestrationRun
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
from huddleroom.models.orchestration_memory import OrchestrationMemorySection
from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
from huddleroom.services.orchestration_process_service import OrchestrationProcessService


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
    # now gates on run.phase == "authorized" (Task 3). Authorize directly
    # since Start (Task 4) is not built yet.
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
    service, _goal, run = completion_ready_goal
    await stub_tick_baseline(monkeypatch)

    result = await service.tick(db_session, run.id)

    assert result["goal_closeout_process"]["status"] == "waiting_decision"
    assert result["goal_closeout_process"]["decision_id"] is not None
    assert result["run_completed"] is False
    assert result["completion_action_id"] is None


@pytest.mark.asyncio
async def test_approved_closeout_completes_run_in_same_tick(
    db_session, completion_ready_goal, test_user, monkeypatch
):
    service, _goal, run = completion_ready_goal
    await stub_tick_baseline(monkeypatch)
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

    result = await service.tick(db_session, run.id)

    assert result["goal_closeout_process"]["status"] == "completed"
    assert result["goal_closeout_process"]["completion_authorized"] is True
    assert result["run_completed"] is True
    assert result["completion_action_id"] is not None


@pytest.mark.asyncio
async def test_trivial_closeout_and_completion_finish_in_one_tick(
    db_session, trivial_completion_ready_goal, monkeypatch
):
    service, _goal, run = trivial_completion_ready_goal
    await stub_tick_baseline(monkeypatch)

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

    await service.tick(db_session, run.id)

    assert run.status != "completed"
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    assert str(item.id) in {f["id"] for f in situation.untracked_follow_ups}
