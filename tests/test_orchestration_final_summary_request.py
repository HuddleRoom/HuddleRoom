import json
import uuid

# pylint: disable=not-callable
import pytest
import pytest_asyncio
from fastapi import HTTPException
from sqlalchemy import func, select, update

pytestmark = pytest.mark.usefixtures(
    "safe_goal_analysis", "safe_agent_definition_review", "safe_effectiveness_review_continue"
)

from huddleroom.models.agent import Agent
from huddleroom.models.artifact import Artifact
from huddleroom.models.event_log import EventLog
from huddleroom.models.orchestration import (
    OrchestrationAction,
    OrchestrationAgentSuggestion,
    OrchestrationEvidence,
    OrchestrationGate,
    OrchestrationRun,
)
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.schemas.orchestration import OrchestrationGoalCreate
from huddleroom.services.event_bus import emit_event_once
from huddleroom.services.orchestration_service import OrchestrationService
from tests.conftest import heal_baseline_drift_for_test


@pytest.fixture(autouse=True)
def no_live_decisions(stub_decision):
    stub_decision(lambda _context: {"action_type": "noop"})


@pytest_asyncio.fixture(autouse=True)
async def runnable_workspace(db_session, test_project, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    test_project.workspace_path = str(workspace.resolve())
    await db_session.flush()


def _agent(name: str, role: str, capabilities: list[str]) -> Agent:
    return Agent(
        name=f"{name}-{uuid.uuid4()}",
        role=role,
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=capabilities,
        config={},
        is_active=True,
    )


async def _complete_goal_definition(db_session, goal, run):
    from tests.conftest import complete_baseline_processes, heal_baseline_drift_for_test

    run.baseline_authorized = True
    await complete_baseline_processes(db_session, goal, run)
    # This whole file exercises completion machinery downstream of
    # baseline_ready (final summary / closeout / complete_run), which tick()
    # now gates on run.phase == "authorized" (Task 3). Authorize directly
    # since Start (Task 4) is not built yet.
    run.phase = "authorized"
    await db_session.flush()
    # This file's tests routinely mutate goal/roster state (new agents,
    # weight) between baseline completion and re-invoking this helper --
    # unrelated to baseline staleness itself, but item 3 now converts any
    # resulting drift into a one-time suggestion instead of silently
    # auto-rerunning, which would otherwise 409 every downstream call here.
    await heal_baseline_drift_for_test(db_session, goal, run)


async def _authorize_completion_closeout(db_session, service, goal, run):
    from huddleroom.services.orchestration_goal_closeout import GoalCloseoutProcess

    goal.weight = "trivial"
    await heal_baseline_drift_for_test(db_session, goal, run)
    result = await GoalCloseoutProcess().advance(
        db_session,
        goal,
        run,
        preconditions=await service._closeout_preconditions_manifest(db_session, goal, run),
    )
    assert result["completion_authorized"] is True


async def _make_run(db_session, project_id):
    service = OrchestrationService()
    goal, run = await service.create_goal(
        db_session,
        project_id=project_id,
        data=OrchestrationGoalCreate(
            objective="Deliver an evidence-backed release",
            success_criteria=[
                {"key": "implemented", "description": "The requested change is complete."},
                {"key": "validated", "description": "Independent validation passed."},
            ],
            constraints={},
            budget={},
        ),
        created_by_user_id=None,
    )
    await _complete_goal_definition(db_session, goal, run)
    return service, goal, run


async def _accepted_gate(db_session, run_id, key: str):
    gate = OrchestrationGate(
        run_id=run_id,
        success_criterion_key=f"plan_item:{key}",
        gate_type="work_completed",
        required_evidence={"success_criterion_keys": [key]},
        status="accepted",
    )
    db_session.add(gate)
    await db_session.flush()
    evidence = OrchestrationEvidence(
        run_id=run_id,
        gate_id=gate.id,
        source_type="verification",
        source_id=uuid.uuid4(),
        observed_event_id=None,
        producer_agent_id=None,
        verdict="accepted",
        evidence_metadata={"fixture": "final-summary"},
    )
    db_session.add(evidence)
    await db_session.flush()
    return gate, evidence


async def _real_planning_expansion_run(db_session, project_id, planner: Agent, developer: Agent):
    service = OrchestrationService()
    goal, run = await service.create_goal(
        db_session,
        project_id=project_id,
        data=OrchestrationGoalCreate(
            objective="Ship the accepted plan items and validate the outcome",
            success_criteria=[
                {"key": "expanded", "description": "Accepted plan items become delegated work."},
                {"key": "validated", "description": "Independent validation passed."},
            ],
            constraints={},
            budget={},
        ),
        created_by_user_id=None,
    )
    await _complete_goal_definition(db_session, goal, run)
    request_action = await service.execute_request_plan_action(
        db_session,
        run_id=run.id,
        request={
            "action_type": "request_plan",
            "agent_id": str(planner.id),
            "work_function": "planning",
            "scope": "Create a plan with independently verifiable items.",
        },
        idempotency_key=f"run:{run.id}:kind:request_plan",
    )
    artifact = Artifact(
        project_id=project_id,
        name="upstream-plan",
        artifact_type="plan",
        status="draft",
        linked_task_id=request_action.target_id,
        created_by_agent=planner.id,
        metadata_={
            "kind": "implementation_plan",
            "plan_items": [
                {
                    "id": "implement-api",
                    "title": "Implement API endpoint",
                    "work_function": "implementation",
                    "agent_id": str(developer.id),
                    "scope": "Implement the accepted plan item without changing unrelated files.",
                    "deliverable": "A code change plus focused pytest output.",
                    "inputs": ["ROADMAP.md M7 Phase 12"],
                    "success_evidence": ["pytest output for focused implementation checks"],
                    "required_evidence": {"required_source_types": ["task"], "min_count": 1},
                    "success_criterion_keys": ["expanded", "validated"],
                }
            ],
        },
    )
    db_session.add(artifact)
    await db_session.flush()
    planning_task = await db_session.get(Task, request_action.target_id)
    planning_task.status = "in_progress"
    planning_session = Session(
        agent_id=planner.id,
        task_id=planning_task.id,
        project_id=project_id,
        adapter_type="api",
        status="completed",
        output=json.dumps({"status": "done", "plan": "Submitted the linked plan artifact."}),
        metadata_={"orchestration": dict(planning_task.metadata_["orchestration"])},
        origin="auto",
    )
    db_session.add(planning_session)
    await db_session.flush()
    from huddleroom.services.session_sync import sync_task_from_session

    await sync_task_from_session(db_session, planning_session)
    await emit_event_once(
        db_session,
        project_id,
        "session.completed",
        {"session_id": str(planning_session.id), "task_id": str(planning_task.id)},
        dedup_key=f"phase17-real-flow-plan-session:{planning_session.id}",
    )
    await service.execute_accept_plan_action(
        db_session,
        run_id=run.id,
        request={"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
        idempotency_key=f"run:{run.id}:kind:accept_plan",
    )
    release = await service.tick(db_session, run.id)
    assert release["authorized_execution"] == {"step": "release_work", "released": 1}
    expansion_action = (
        await db_session.execute(
            select(OrchestrationAction).where(
                OrchestrationAction.run_id == run.id,
                OrchestrationAction.action_type == "expand_plan_item",
                OrchestrationAction.status == "completed",
            )
        )
    ).scalar_one()
    expanded_task = await db_session.get(Task, expansion_action.target_id)
    expanded_gate = await db_session.get(
        OrchestrationGate,
        uuid.UUID(expanded_task.metadata_["orchestration"]["plan_item_gate_id"]),
    )
    expanded_task.status = "in_progress"
    producer_session = Session(
        agent_id=developer.id,
        task_id=expanded_task.id,
        project_id=project_id,
        adapter_type="api",
        status="completed",
        output=json.dumps({"status": "done", "changes": ["Implemented the accepted plan item."]}),
        metadata_={"orchestration": dict(expanded_task.metadata_["orchestration"])},
        origin="auto",
    )
    db_session.add(producer_session)
    await db_session.flush()
    await sync_task_from_session(db_session, producer_session)
    await emit_event_once(
        db_session,
        project_id,
        "session.completed",
        {"session_id": str(producer_session.id), "task_id": str(expanded_task.id)},
        dedup_key=f"phase17-real-flow-producer-session:{producer_session.id}",
    )
    verification_action = await service.execute_request_verification_action(
        db_session,
        run.id,
        {
            "action_type": "request_verification",
            "gate_id": str(expanded_gate.id),
            "work_function": "validation",
        },
        f"run:{run.id}:kind:final-summary-verification",
    )
    verification_task = await db_session.get(Task, verification_action.target_id)
    assert verification_task is not None and verification_task.assigned_to == planner.id
    verification_task.status = "in_progress"
    verification_session = Session(
        agent_id=planner.id,
        task_id=verification_task.id,
        project_id=project_id,
        adapter_type="api",
        status="completed",
        output=json.dumps({"status": "done", "verdict": "accepted", "evidence": ["Independent verification passed."]}),
        metadata_={"orchestration": dict(verification_task.metadata_["orchestration"])},
        origin="auto",
    )
    db_session.add(verification_session)
    await db_session.flush()
    await sync_task_from_session(db_session, verification_session)
    await emit_event_once(
        db_session,
        project_id,
        "session.completed",
        {"session_id": str(verification_session.id), "task_id": str(verification_task.id)},
        dedup_key=f"phase17-real-flow-verification-session:{verification_session.id}",
    )
    events = await service._new_events(db_session, project_id, run.event_cursor)
    assert await service._ingest_evidence_from_events(db_session, run, events) >= 2
    assert await service.validate_open_gates(db_session, run.id) >= 1
    await db_session.refresh(expanded_gate)
    assert expanded_gate.status == "accepted"
    return service, goal, run


async def _events(db_session, project_id, event_type: str):
    result = await db_session.execute(
        select(EventLog)
        .where(EventLog.project_id == project_id, EventLog.event_type == event_type)
        .order_by(EventLog.seq.asc())
    )
    return list(result.scalars().all())


async def _valid_summary_output(db_session, goal, run) -> str:
    evidence_by_criterion = {}
    for gate, evidence_id in (
        await db_session.execute(
            select(OrchestrationGate, OrchestrationEvidence.id)
            .join(OrchestrationEvidence, OrchestrationEvidence.gate_id == OrchestrationGate.id)
            .where(OrchestrationEvidence.run_id == run.id, OrchestrationEvidence.verdict == "accepted",
                   OrchestrationEvidence.source_type == "verification")
        )
    ).all():
        for key in gate.required_evidence.get("success_criterion_keys", []):
            evidence_by_criterion.setdefault(key, evidence_id)
    assert evidence_by_criterion
    return json.dumps(
        {
            "summary": "Evidence-backed final summary.",
            "criteria": [
                {
                    "criterion_key": criterion["key"],
                    "evidence_ids": [str(evidence_by_criterion[criterion["key"]])],
                }
                for index, criterion in enumerate(goal.success_criteria)
            ],
            "unresolved_gaps": [],
        }
    )


@pytest.mark.asyncio
async def test_request_final_summary_routes_to_best_fit_and_replays_once(db_session, test_project):
    developer = _agent("developer", "developer", ["implementation"])
    writer = _agent("writer", "technical writer", ["summarization", "documentation"])
    db_session.add_all([developer, writer])
    await db_session.flush()
    service, goal, run = await _make_run(db_session, test_project.id)
    implemented_gate, implemented_evidence = await _accepted_gate(
        db_session, run.id, "implemented"
    )
    validated_gate, validated_evidence = await _accepted_gate(
        db_session, run.id, "validated"
    )
    producer_claim = OrchestrationEvidence(
            run_id=run.id,
            gate_id=implemented_gate.id,
            source_type="task",
            source_id=uuid.uuid4(),
            verdict="accepted",
            evidence_metadata={"fixture": "producer-claim"},
    )
    db_session.add(producer_claim)
    await db_session.flush()
    key = f"run:{run.id}:kind:request_final_summary"
    request = {"action_type": "request_final_summary", "work_function": "summarization"}

    first = await service.execute_request_final_summary_action(
        db_session, run.id, request, key
    )
    second = await service.execute_request_final_summary_action(
        db_session, run.id, request, key
    )

    assert first.id == second.id
    assert first.status == "completed"
    assert first.target_type == "task"
    task = await db_session.get(Task, first.target_id)
    assert task is not None
    assert task.assigned_to == writer.id
    assert task.metadata_["orchestration"]["work_function"] == "summarization"
    assert task.metadata_["orchestration"]["final_summary"] is True
    gate_id = uuid.UUID(task.metadata_["orchestration"]["gate_id"])
    gate = await db_session.get(OrchestrationGate, gate_id)
    assert gate.gate_type == "final_summary_accepted"
    assert gate.status == "open"
    assert gate.required_evidence == {"required_source_types": ["session"], "min_count": 1}
    summary_metadata = task.metadata_["orchestration_final_summary"]
    assert summary_metadata["accepted_gate_ids"] == [
        str(implemented_gate.id),
        str(validated_gate.id),
    ]
    assert {uuid.UUID(value) for value in summary_metadata["accepted_evidence_ids"]} == {
        implemented_evidence.id,
        validated_evidence.id,
        producer_claim.id,
    }
    delegation = await db_session.scalar(
        select(OrchestrationAction).where(
            OrchestrationAction.run_id == run.id,
            OrchestrationAction.action_type == "create_delegation_task",
            OrchestrationAction.target_id == task.id,
        )
    )
    criterion_manifest = json.loads(next(
        value.removeprefix("Criterion-scoped accepted verification evidence: ")
        for value in delegation.request["inputs"]
        if value.startswith("Criterion-scoped accepted verification evidence: ")
    ))
    assert criterion_manifest == {
        "implemented": [str(implemented_evidence.id)],
        "validated": [str(validated_evidence.id)],
    }
    assert await db_session.scalar(
        select(func.count(OrchestrationGate.id)).where(
            OrchestrationGate.run_id == run.id,
            OrchestrationGate.gate_type == "final_summary_accepted",
        )
    ) == 1
    assert await db_session.scalar(
        select(func.count(Task.id)).where(
            Task.metadata_["orchestration"]["final_summary"].as_boolean().is_(True)
        )
    ) == 1
    events = await _events(db_session, test_project.id, "orchestration.final_summary_requested")
    assert len(events) == 1
    assert events[0].payload["goal_id"] == str(goal.id)
    assert events[0].payload["gate_id"] == str(gate.id)


@pytest.mark.asyncio
async def test_request_final_summary_requires_accepted_evidence_for_every_non_summary_gate(
    db_session,
    test_project,
):
    writer = _agent("writer", "summarizer", ["summarization"])
    db_session.add(writer)
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, test_project.id)
    await _accepted_gate(db_session, run.id, "implemented")
    db_session.add(
        OrchestrationGate(
            run_id=run.id,
            success_criterion_key="validation-work",
            gate_type="validation_passed",
            required_evidence={"required_source_types": ["session"], "min_count": 1},
            status="accepted",
        )
    )
    await db_session.flush()

    with pytest.raises(HTTPException) as exc:
        await service.execute_request_final_summary_action(
            db_session,
            run.id,
            {"action_type": "request_final_summary", "work_function": "summarization"},
            f"run:{run.id}:kind:request_final_summary",
        )

    assert exc.value.status_code == 409
    assert exc.value.detail == "All non-summary gates must be accepted and have accepted evidence"
    assert await db_session.scalar(
        select(func.count(Task.id)).where(
            Task.metadata_["orchestration"]["run_id"].as_string() == str(run.id),
            Task.metadata_["orchestration"]["final_summary"].as_boolean().is_(True),
        )
    ) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_status", ["paused", "cancelled"])
async def test_request_final_summary_rejects_terminal_goal(
    db_session,
    test_project,
    terminal_status,
):
    writer = _agent("writer", "summarizer", ["summarization"])
    db_session.add(writer)
    await db_session.flush()
    service, goal, run = await _make_run(db_session, test_project.id)
    await _accepted_gate(db_session, run.id, "implemented")
    await _accepted_gate(db_session, run.id, "validated")
    goal.status = terminal_status
    run.status = "running"
    await db_session.flush()

    with pytest.raises(HTTPException) as exc:
        await service.execute_request_final_summary_action(
            db_session,
            run.id,
            {"action_type": "request_final_summary", "work_function": "summarization"},
            f"run:{run.id}:kind:request_final_summary",
        )

    assert exc.value.status_code == 409
    assert exc.value.detail == f"Orchestration goal is {terminal_status}"
    assert await db_session.scalar(
        select(func.count(OrchestrationAction.id)).where(
            OrchestrationAction.run_id == run.id,
            OrchestrationAction.action_type == "request_final_summary",
        )
    ) == 0
    assert await db_session.scalar(
        select(func.count(OrchestrationGate.id)).where(
            OrchestrationGate.run_id == run.id,
            OrchestrationGate.gate_type == "final_summary_accepted",
        )
    ) == 0
    assert await db_session.scalar(
        select(func.count(Task.id)).where(
            Task.metadata_["orchestration"]["run_id"].as_string() == str(run.id),
            Task.metadata_["orchestration"]["final_summary"].as_boolean().is_(True),
        )
    ) == 0


@pytest.mark.asyncio
async def test_request_final_summary_accepts_real_plan_acceptance_and_plan_item_gate_keys(
    db_session,
    test_project,
):
    planner = _agent("planner", "planner", ["planning", "validation"])
    developer = _agent("developer", "developer", ["implementation"])
    writer = _agent("writer", "technical writer", ["summarization"])
    db_session.add_all([planner, developer, writer])
    await db_session.flush()
    service, _goal, run = await _real_planning_expansion_run(
        db_session,
        test_project.id,
        planner,
        developer,
    )

    action = await service.execute_request_final_summary_action(
        db_session,
        run.id,
        {"action_type": "request_final_summary", "work_function": "summarization"},
        f"run:{run.id}:kind:request_final_summary",
    )

    task = await db_session.get(Task, action.target_id)
    summary_metadata = task.metadata_["orchestration_final_summary"]
    accepted_gate_ids = {uuid.UUID(value) for value in summary_metadata["accepted_gate_ids"]}
    accepted_evidence_ids = {uuid.UUID(value) for value in summary_metadata["accepted_evidence_ids"]}
    linked_gates = [await db_session.get(OrchestrationGate, gate_id) for gate_id in accepted_gate_ids]

    assert any(gate.success_criterion_key == "plan" for gate in linked_gates if gate is not None)
    assert any(
        gate.success_criterion_key == "plan_item:implement-api"
        for gate in linked_gates
        if gate is not None
    )


@pytest.mark.asyncio
async def test_request_final_summary_accepts_human_override_evidence(db_session, test_project):
    writer = _agent("writer", "technical writer", ["summarization"])
    db_session.add(writer)
    await db_session.flush()
    service, goal, run = await _make_run(db_session, test_project.id)
    overridden_gate, _ = await _accepted_gate(db_session, run.id, "implemented")
    gate_to_override = OrchestrationGate(
        run_id=run.id,
        success_criterion_key="validated",
        gate_type="work_completed",
        required_evidence={"required_source_types": ["task"], "min_count": 1},
        status="failed",
        failure_reason="Independent validation did not pass",
    )
    db_session.add(gate_to_override)
    await db_session.flush()
    await service.override_gate(
        db_session,
        test_project.id,
        goal.id,
        gate_id=gate_to_override.id,
        decision="accept",
        reason="Human reviewed the validation evidence and accepts the gate for this run.",
        user_id=None,
    )

    action = await service.execute_request_final_summary_action(
        db_session,
        run.id,
        {"action_type": "request_final_summary", "work_function": "summarization"},
        f"run:{run.id}:kind:request_final_summary",
    )

    task = await db_session.get(Task, action.target_id)
    summary_metadata = task.metadata_["orchestration_final_summary"]
    accepted_gate_ids = {uuid.UUID(value) for value in summary_metadata["accepted_gate_ids"]}
    accepted_evidence_ids = {uuid.UUID(value) for value in summary_metadata["accepted_evidence_ids"]}
    override_evidence = (
        await db_session.execute(
            select(OrchestrationEvidence).where(
                OrchestrationEvidence.gate_id == gate_to_override.id,
                OrchestrationEvidence.source_type == "human_override",
            )
        )
    ).scalar_one()

    assert overridden_gate.id in accepted_gate_ids
    assert gate_to_override.id in accepted_gate_ids
    assert override_evidence.id in accepted_evidence_ids


@pytest.mark.asyncio
async def test_request_final_summary_weak_fit_records_suggestion(db_session, test_project):
    developer = _agent("developer", "developer", ["implementation"])
    db_session.add(developer)
    await db_session.flush()
    service, goal, run = await _make_run(db_session, test_project.id)
    await _accepted_gate(db_session, run.id, "implemented")
    await _accepted_gate(db_session, run.id, "validated")

    with pytest.raises(HTTPException) as exc:
        await service.execute_request_final_summary_action(
            db_session,
            run.id,
            {"action_type": "request_final_summary", "work_function": "summarization"},
            f"run:{run.id}:kind:request_final_summary",
        )

    assert exc.value.status_code == 409
    assert exc.value.detail == "No strong summarization agent fit"
    suggestion = (
        await db_session.execute(
            select(OrchestrationAgentSuggestion).where(
                OrchestrationAgentSuggestion.run_id == run.id,
                OrchestrationAgentSuggestion.missing_work_function == "summarization",
            )
        )
    ).scalar_one()
    assert suggestion.status == "open"
    assert await db_session.scalar(
        select(func.count(OrchestrationGate.id)).where(
            OrchestrationGate.run_id == run.id,
            OrchestrationGate.gate_type == "final_summary_accepted",
        )
    ) == 0
    assert await db_session.scalar(
        select(func.count(OrchestrationAction.id)).where(
            OrchestrationAction.run_id == run.id,
            OrchestrationAction.action_type == "request_final_summary",
        )
    ) == 0

    writer = _agent("writer", "summarizer", ["summarization"])
    db_session.add(writer)
    await db_session.flush()
    await _complete_goal_definition(db_session, goal, run)
    retried = await service.execute_request_final_summary_action(
        db_session,
        run.id,
        {"action_type": "request_final_summary", "work_function": "summarization"},
        f"run:{run.id}:kind:request_final_summary",
    )
    task = await db_session.get(Task, retried.target_id)
    assert retried.status == "completed"
    assert task.assigned_to == writer.id


async def _request_summary(db_session, test_project):
    writer = _agent("writer", "technical writer", ["summarization", "documentation"])
    db_session.add(writer)
    await db_session.flush()
    service, goal, run = await _make_run(db_session, test_project.id)
    await _accepted_gate(db_session, run.id, "implemented")
    await _accepted_gate(db_session, run.id, "validated")
    tick_result = await service.tick(db_session, run.id)
    assert tick_result["final_summary_action_id"] is not None
    action = await db_session.get(OrchestrationAction, tick_result["final_summary_action_id"])
    task = await db_session.get(Task, action.target_id)
    gate = await db_session.get(
        OrchestrationGate,
        uuid.UUID(task.metadata_["orchestration"]["gate_id"]),
    )
    await _complete_goal_definition(db_session, goal, run)
    return service, goal, run, writer, task, gate


@pytest.mark.asyncio
async def test_tick_requests_summary_after_all_non_summary_gates_pass(db_session, test_project):
    writer = _agent("writer", "technical writer", ["summarization", "documentation"])
    db_session.add(writer)
    await db_session.flush()
    service, goal, run = await _make_run(db_session, test_project.id)
    await _accepted_gate(db_session, run.id, "implemented")
    await _accepted_gate(db_session, run.id, "validated")

    result = await service.tick(db_session, run.id)

    await db_session.refresh(goal)
    await db_session.refresh(run)
    summary_actions = list(
        (
            await db_session.execute(
                select(OrchestrationAction).where(
                    OrchestrationAction.run_id == run.id,
                    OrchestrationAction.action_type == "request_final_summary",
                )
            )
        ).scalars().all()
    )
    assert result["run_completed"] is False
    assert result["final_summary_action_id"] == summary_actions[0].id
    assert len(summary_actions) == 1
    assert summary_actions[0].status == "completed"
    assert goal.status == "active"
    assert run.status == "running"


@pytest.mark.asyncio
async def test_tick_weak_summary_fit_suggests_agent_then_retries(db_session, test_project):
    developer = _agent("developer", "developer", ["implementation"])
    db_session.add(developer)
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, test_project.id)
    await _accepted_gate(db_session, run.id, "implemented")
    await _accepted_gate(db_session, run.id, "validated")

    weak_tick = await service.tick(db_session, run.id)

    assert weak_tick["final_summary_action_id"] is None
    suggestion = (
        await db_session.execute(
            select(OrchestrationAgentSuggestion).where(
                OrchestrationAgentSuggestion.run_id == run.id,
                OrchestrationAgentSuggestion.missing_work_function == "summarization",
            )
        )
    ).scalar_one()
    assert suggestion.status == "open"

    writer = _agent("writer", "summarizer", ["summarization"])
    db_session.add(writer)
    await db_session.flush()
    await _complete_goal_definition(db_session, _goal, run)
    retry_tick = await service.tick(db_session, run.id)

    assert retry_tick["final_summary_action_id"] is not None
    action = await db_session.get(OrchestrationAction, retry_tick["final_summary_action_id"])
    task = await db_session.get(Task, action.target_id)
    assert action.status == "completed"
    assert task.assigned_to == writer.id


@pytest.mark.asyncio
async def test_summary_task_done_without_session_keeps_gate_and_run_open(db_session, test_project):
    service, goal, run, _writer, task, gate = await _request_summary(db_session, test_project)
    task.status = "done"
    await emit_event_once(
        db_session,
        test_project.id,
        "task.status_changed",
        {"task_id": str(task.id), "status": "done", "previous_status": "in_progress"},
        dedup_key=f"phase17-task-done:{task.id}",
    )

    result = await service.tick(db_session, run.id)

    await db_session.refresh(goal)
    await db_session.refresh(run)
    await db_session.refresh(gate)
    assert result["run_completed"] is False
    assert gate.status == "open"
    assert gate.failure_reason == "Missing evidence: session"
    assert goal.status == "active"
    assert run.status == "running"


@pytest.mark.asyncio
async def test_empty_summary_session_fails_final_gate(db_session, test_project):
    service, _goal, run, writer, task, gate = await _request_summary(db_session, test_project)
    session = Session(
        agent_id=writer.id,
        task_id=task.id,
        project_id=test_project.id,
        adapter_type="api",
        status="completed",
        output="   ",
        metadata_={},
        origin="auto",
    )
    db_session.add(session)
    await db_session.flush()
    await emit_event_once(
        db_session,
        test_project.id,
        "session.completed",
        {"session_id": str(session.id), "task_id": str(task.id)},
        dedup_key=f"phase17-session-completed:{session.id}",
    )

    await service.tick(db_session, run.id)

    await db_session.refresh(gate)
    assert gate.status == "failed"
    assert gate.failure_reason == "Final summary output is missing"


@pytest.mark.asyncio
async def test_summary_session_requires_valid_json(db_session, test_project):
    service, _goal, run, writer, task, gate = await _request_summary(db_session, test_project)
    session = Session(
        agent_id=writer.id,
        task_id=task.id,
        project_id=test_project.id,
        adapter_type="api",
        status="completed",
        output="This is prose, not the required JSON envelope.",
        metadata_={},
        origin="auto",
    )
    db_session.add(session)
    await db_session.flush()
    await emit_event_once(
        db_session,
        test_project.id,
        "session.completed",
        {"session_id": str(session.id), "task_id": str(task.id)},
        dedup_key=f"phase17-invalid-json:{session.id}",
    )

    result = await service.tick(db_session, run.id)

    await db_session.refresh(gate)
    assert result["run_completed"] is False
    assert gate.status == "failed"
    assert gate.failure_reason == "Final summary output must be valid JSON"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "expected_reason"),
    [
        ("root_type", "Final summary output must be a JSON object"),
        ("summary_type", "Final summary text must be a non-empty string"),
        ("summary_blank", "Final summary text must be a non-empty string"),
        ("criteria_type", "Final summary criteria must be a list"),
        ("criterion_key_type", "Final summary criterion_key must be a non-empty string"),
        ("criterion_key_blank", "Final summary criterion_key must be a non-empty string"),
        (
            "criterion_key_whitespace",
            "Final summary criterion_key must not contain surrounding whitespace",
        ),
        ("declared_criterion_key_type", "Declared success criteria keys are invalid"),
        (
            "declared_criterion_key_whitespace",
            "Declared success criteria keys must not contain surrounding whitespace",
        ),
        ("evidence_ids_type", "Final summary evidence_ids must be a non-empty list"),
        ("extra_criterion", "Final summary criterion coverage is incomplete"),
        ("duplicate_criterion", "Final summary criterion mapping contains duplicate criterion_key"),
        ("missing_criterion", "Final summary criterion coverage is incomplete"),
    ],
)
async def test_summary_session_rejects_invalid_json_shape_and_coverage(
    db_session,
    test_project,
    case,
    expected_reason,
):
    service, goal, run, writer, task, gate = await _request_summary(db_session, test_project)
    accepted_evidence_id = str(
        await db_session.scalar(
            select(OrchestrationEvidence.id).where(
                OrchestrationEvidence.run_id == run.id,
                OrchestrationEvidence.verdict == "accepted",
            )
        )
    )
    valid_implemented = {"criterion_key": "implemented", "evidence_ids": [accepted_evidence_id]}
    valid_validated = {"criterion_key": "validated", "evidence_ids": [accepted_evidence_id]}
    payloads = {
        "root_type": ["not", "an", "object"],
        "summary_type": {"summary": 123, "criteria": [], "unresolved_gaps": []},
        "summary_blank": {"summary": "   ", "criteria": [], "unresolved_gaps": []},
        "criteria_type": {"summary": "Summary", "criteria": {}, "unresolved_gaps": []},
        "criterion_key_type": {
            "summary": "Summary",
            "criteria": [{"criterion_key": 123, "evidence_ids": [accepted_evidence_id]}],
            "unresolved_gaps": [],
        },
        "criterion_key_blank": {
            "summary": "Summary",
            "criteria": [{"criterion_key": "   ", "evidence_ids": [accepted_evidence_id]}],
            "unresolved_gaps": [],
        },
        "criterion_key_whitespace": {
            "summary": "Summary",
            "criteria": [{"criterion_key": " implemented ", "evidence_ids": [accepted_evidence_id]}],
            "unresolved_gaps": [],
        },
        "declared_criterion_key_type": {
            "summary": "Summary",
            "criteria": [valid_implemented, valid_validated],
            "unresolved_gaps": [],
        },
        "declared_criterion_key_whitespace": {
            "summary": "Summary",
            "criteria": [valid_implemented, valid_validated],
            "unresolved_gaps": [],
        },
        "evidence_ids_type": {
            "summary": "Summary",
            "criteria": [{"criterion_key": "implemented", "evidence_ids": accepted_evidence_id}],
            "unresolved_gaps": [],
        },
        "extra_criterion": {
            "summary": "Summary",
            "criteria": [
                valid_implemented,
                valid_validated,
                {"criterion_key": "undeclared", "evidence_ids": [accepted_evidence_id]},
            ],
            "unresolved_gaps": [],
        },
        "duplicate_criterion": {
            "summary": "Summary",
            "criteria": [valid_implemented, valid_implemented, valid_validated],
            "unresolved_gaps": [],
        },
        "missing_criterion": {
            "summary": "Only one declared criterion is covered.",
            "criteria": [
                {"criterion_key": "implemented", "evidence_ids": [accepted_evidence_id]}
            ],
            "unresolved_gaps": [],
        },
    }
    if case == "declared_criterion_key_type":
        goal.success_criteria = [
            {"key": 123, "description": "Invalid non-string key."},
            {"key": "validated", "description": "Independent validation passed."},
        ]
        await heal_baseline_drift_for_test(db_session, goal, run)
    if case == "declared_criterion_key_whitespace":
        goal.success_criteria = [
            {"key": " implemented ", "description": "Invalid surrounding whitespace."},
            {"key": "validated", "description": "Independent validation passed."},
        ]
        await heal_baseline_drift_for_test(db_session, goal, run)
    session = Session(
        agent_id=writer.id,
        task_id=task.id,
        project_id=test_project.id,
        adapter_type="api",
        status="completed",
        output=json.dumps(payloads[case]),
        metadata_={},
        origin="auto",
    )
    db_session.add(session)
    await db_session.flush()
    await emit_event_once(
        db_session,
        test_project.id,
        "session.completed",
        {"session_id": str(session.id), "task_id": str(task.id)},
        dedup_key=f"phase17-invalid-json-shape:{case}:{session.id}",
    )

    result = await service.tick(db_session, run.id)

    await db_session.refresh(gate)
    assert result["run_completed"] is False
    assert gate.status == "failed"
    assert gate.failure_reason == expected_reason


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "wrong_agent", "expected_reason"),
    [
        ("running", False, "Final summary session is not completed"),
        ("completed", True, "Final summary session attribution is invalid"),
    ],
)
async def test_summary_session_requires_completed_assigned_agent(
    db_session,
    test_project,
    status,
    wrong_agent,
    expected_reason,
):
    service, goal, run, writer, task, gate = await _request_summary(db_session, test_project)
    session_agent = writer
    if wrong_agent:
        session_agent = _agent("other", "technical writer", ["summarization"])
        db_session.add(session_agent)
        await db_session.flush()
        await heal_baseline_drift_for_test(db_session, goal, run)
    session = Session(
        agent_id=session_agent.id,
        task_id=task.id,
        project_id=test_project.id,
        adapter_type="api",
        status=status,
        output=await _valid_summary_output(db_session, goal, run),
        metadata_={},
        origin="auto",
    )
    db_session.add(session)
    await db_session.flush()
    await emit_event_once(
        db_session,
        test_project.id,
        "session.completed",
        {"session_id": str(session.id), "task_id": str(task.id)},
        dedup_key=f"phase17-session-completed:{session.id}",
    )

    await service.tick(db_session, run.id)

    await db_session.refresh(gate)
    assert gate.status == "failed"
    assert gate.failure_reason == expected_reason


@pytest.mark.asyncio
async def test_summary_session_rejects_unaccepted_evidence_id(db_session, test_project):
    service, _goal, run, writer, task, gate = await _request_summary(db_session, test_project)
    unaccepted_id = uuid.uuid4()
    session = Session(
        agent_id=writer.id,
        task_id=task.id,
        project_id=test_project.id,
        adapter_type="api",
        status="completed",
        output=json.dumps(
            {
                "summary": "Coverage cites evidence that is not accepted for this run.",
                "criteria": [
                    {"criterion_key": "implemented", "evidence_ids": [str(unaccepted_id)]},
                    {"criterion_key": "validated", "evidence_ids": [str(unaccepted_id)]},
                ],
                "unresolved_gaps": [],
            }
        ),
        metadata_={},
        origin="auto",
    )
    db_session.add(session)
    await db_session.flush()
    await emit_event_once(
        db_session,
        test_project.id,
        "session.completed",
        {"session_id": str(session.id), "task_id": str(task.id)},
        dedup_key=f"phase17-unaccepted-evidence:{session.id}",
    )

    result = await service.tick(db_session, run.id)

    await db_session.refresh(gate)
    assert result["run_completed"] is False
    assert gate.status == "failed"
    assert gate.failure_reason == "Final summary evidence must be accepted verification linked to its criterion"


@pytest.mark.asyncio
async def test_summary_rejects_reusing_one_criterion_verification_for_another(db_session, test_project):
    service, goal, run, writer, task, gate = await _request_summary(db_session, test_project)
    implemented_evidence = await db_session.scalar(
        select(OrchestrationEvidence).join(OrchestrationGate).where(
            OrchestrationEvidence.run_id == run.id,
            OrchestrationEvidence.source_type == "verification",
            OrchestrationGate.success_criterion_key == "plan_item:implemented",
        )
    )
    assert implemented_evidence is not None
    session = Session(
        agent_id=writer.id,
        task_id=task.id,
        project_id=test_project.id,
        adapter_type="api",
        status="completed",
        output=json.dumps({
            "summary": "Incorrectly reuses implementation proof for validation.",
            "criteria": [
                {"criterion_key": "implemented", "evidence_ids": [str(implemented_evidence.id)]},
                {"criterion_key": "validated", "evidence_ids": [str(implemented_evidence.id)]},
            ],
            "unresolved_gaps": [],
        }),
        metadata_={},
        origin="auto",
    )
    db_session.add(session)
    await db_session.flush()
    await emit_event_once(
        db_session, test_project.id, "session.completed",
        {"session_id": str(session.id), "task_id": str(task.id)},
        dedup_key=f"phase17-reused-criterion-evidence:{session.id}",
    )

    result = await service.tick(db_session, run.id)

    await db_session.refresh(goal)
    await db_session.refresh(run)
    await db_session.refresh(gate)
    assert result["run_completed"] is False
    assert goal.status != "completed"
    assert run.status != "completed"
    assert gate.failure_reason == "Final summary evidence must be accepted verification linked to its criterion"


@pytest.mark.asyncio
async def test_summary_allows_one_plan_item_verification_linked_to_both_criteria(db_session, test_project):
    writer = _agent("writer", "technical writer", ["summarization"])
    db_session.add(writer)
    await db_session.flush()
    service, goal, run = await _make_run(db_session, test_project.id)
    goal.weight = "trivial"
    await heal_baseline_drift_for_test(db_session, goal, run)
    work_gate = OrchestrationGate(
        run_id=run.id,
        success_criterion_key="plan_item:joint-delivery",
        gate_type="work_completed",
        required_evidence={"success_criterion_keys": ["implemented", "validated"]},
        status="accepted",
    )
    db_session.add(work_gate)
    await db_session.flush()
    verification = OrchestrationEvidence(
        run_id=run.id,
        gate_id=work_gate.id,
        source_type="verification",
        source_id=uuid.uuid4(),
        verdict="accepted",
        evidence_metadata={"fixture": "joint-criteria"},
    )
    db_session.add(verification)
    await db_session.flush()
    await _complete_goal_definition(db_session, goal, run)
    request = await service.execute_request_final_summary_action(
        db_session, run.id,
        {"action_type": "request_final_summary", "work_function": "summarization"},
        f"run:{run.id}:kind:request_final_summary",
    )
    task = await db_session.get(Task, request.target_id)
    gate = await db_session.get(OrchestrationGate, uuid.UUID(task.metadata_["orchestration"]["gate_id"]))
    session = Session(
        agent_id=writer.id, task_id=task.id, project_id=test_project.id, adapter_type="api", status="completed",
        output=json.dumps({
            "summary": "One independently verified work item satisfies both declared outcomes.",
            "criteria": [
                {"criterion_key": "implemented", "evidence_ids": [str(verification.id)]},
                {"criterion_key": "validated", "evidence_ids": [str(verification.id)]},
            ],
            "unresolved_gaps": [],
        }), metadata_={}, origin="auto",
    )
    db_session.add(session)
    await db_session.flush()
    await emit_event_once(
        db_session, test_project.id, "session.completed",
        {"session_id": str(session.id), "task_id": str(task.id)},
        dedup_key=f"phase17-joint-criterion-summary:{session.id}",
    )

    result = await service.tick(db_session, run.id)

    await db_session.refresh(goal)
    await db_session.refresh(run)
    await db_session.refresh(gate)
    assert result["run_completed"] is True
    assert goal.status == run.status == "completed"
    assert gate.status == "accepted"


@pytest.mark.asyncio
async def test_summary_session_completes_run_once_with_evidence_manifest(db_session, test_project):
    service, goal, run, writer, task, gate = await _request_summary(db_session, test_project)
    goal.weight = "trivial"
    await heal_baseline_drift_for_test(db_session, goal, run)
    session = Session(
        agent_id=writer.id,
        task_id=task.id,
        project_id=test_project.id,
        adapter_type="api",
        status="completed",
        output=await _valid_summary_output(db_session, goal, run),
        metadata_={},
        origin="auto",
    )
    db_session.add(session)
    await db_session.flush()
    await emit_event_once(
        db_session,
        test_project.id,
        "session.completed",
        {"session_id": str(session.id), "task_id": str(task.id)},
        dedup_key=f"phase17-session-completed:{session.id}",
    )

    result = await service.tick(db_session, run.id)
    replay = await service.tick(db_session, run.id)

    await db_session.refresh(goal)
    await db_session.refresh(run)
    await db_session.refresh(gate)
    evidence = (
        await db_session.execute(
            select(OrchestrationEvidence).where(
                OrchestrationEvidence.gate_id == gate.id,
                OrchestrationEvidence.source_type == "session",
            )
        )
    ).scalar_one()
    assert result["run_completed"] is True
    assert replay["run_completed"] is False
    assert gate.status == "accepted"
    assert evidence.verdict == "accepted"
    assert evidence.source_id == session.id
    assert goal.status == "completed"
    assert run.status == "completed"
    assert run.completed_at is not None
    assert run.active_blockers == []
    actions = list(
        (
            await db_session.execute(
                select(OrchestrationAction).where(
                    OrchestrationAction.run_id == run.id,
                    OrchestrationAction.action_type == "complete_run",
                )
            )
        ).scalars().all()
    )
    assert len(actions) == 1
    assert actions[0].status == "completed"
    events = await _events(db_session, test_project.id, "orchestration.run_completed")
    assert len(events) == 1
    payload = events[0].payload
    assert payload["final_summary"]["task_id"] == str(task.id)
    assert payload["final_summary"]["session_id"] == str(session.id)
    assert payload["final_summary"]["evidence_id"] == str(evidence.id)
    assert {item["key"] for item in payload["declared_success_criteria"]} == {
        "implemented",
        "validated",
    }
    assert {item["criterion_key"] for item in payload["criterion_evidence"]} == {
        "implemented",
        "validated",
    }
    accepted_evidence_ids = {
        evidence_id
        for item in payload["accepted_non_summary_gates"]
        for evidence_id in item["accepted_evidence_ids"]
    }
    assert all(
        item["evidence_ids"] and set(item["evidence_ids"]).issubset(accepted_evidence_ids)
        for item in payload["criterion_evidence"]
    )
    accepted_groups = payload["accepted_non_summary_gates"]
    assert {item["success_criterion_key"] for item in accepted_groups} == {
        "plan_item:implemented",
        "plan_item:validated",
    }
    for item in accepted_groups:
        linked_gate = await db_session.get(OrchestrationGate, uuid.UUID(item["gate_id"]))
        linked_evidence = [
            await db_session.get(OrchestrationEvidence, uuid.UUID(evidence_id))
            for evidence_id in item["accepted_evidence_ids"]
        ]
        assert linked_gate is not None
        assert linked_gate.success_criterion_key == item["success_criterion_key"]
        assert linked_evidence
        assert {linked_item.gate_id for linked_item in linked_evidence} == {linked_gate.id}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_summary_session_completes_real_planning_expansion_flow(db_session, test_project, test_user):
    planner = _agent("planner", "planner", ["planning", "validation"])
    developer = _agent("developer", "developer", ["implementation"])
    writer = _agent("writer", "technical writer", ["summarization"])
    planner.description = "Create implementation plans and independently validate completed work."
    developer.description = "Implement accepted plan items."
    developer.adapter_type = "cli"
    developer.config = {"cli_runtime": "codex"}
    writer.description = "Analyze accepted evidence for factual accuracy."
    db_session.add_all([planner, developer, writer])
    await db_session.flush()
    service, goal, run = await _real_planning_expansion_run(
        db_session,
        test_project.id,
        planner,
        developer,
    )
    request_action = await service.execute_request_final_summary_action(
        db_session,
        run.id,
        {"action_type": "request_final_summary", "work_function": "summarization"},
        f"run:{run.id}:kind:request_final_summary",
    )
    task = await db_session.get(Task, request_action.target_id)
    gate = await db_session.get(
        OrchestrationGate,
        uuid.UUID(task.metadata_["orchestration"]["gate_id"]),
    )
    session = Session(
        agent_id=writer.id,
        task_id=task.id,
        project_id=test_project.id,
        adapter_type="api",
        status="completed",
        output=await _valid_summary_output(db_session, goal, run),
        metadata_={},
        origin="auto",
    )
    task.status = "in_progress"
    db_session.add(session)
    await db_session.flush()
    from huddleroom.services.session_sync import sync_task_from_session

    await sync_task_from_session(db_session, session)
    await emit_event_once(
        db_session,
        test_project.id,
        "session.completed",
        {"session_id": str(session.id), "task_id": str(task.id)},
        dedup_key=f"phase17-real-flow-session-completed:{session.id}",
    )

    first = await service.tick(db_session, run.id)
    assert first["authorized_execution"]["outcome"] == "closeout_ready"
    second = await service.tick(db_session, run.id)
    assert second["goal_closeout_process"]["status"] == "waiting_decision"
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService

    decision = (await OrchestrationAuthorityDecisionService().list_decisions(
        db_session, goal.id, status="pending"
    ))[0]
    if decision.authority == "human":
        await OrchestrationAuthorityDecisionService().answer_decision(
            db_session,
            decision,
            selected_option="approve_completion",
            reason="Evidence is sufficient.",
            decided_by_user_id=test_user.id,
        )
    else:
        await OrchestrationAuthorityDecisionService().answer_decision(
            db_session,
            decision,
            selected_option="approve_completion",
            reason="Evidence is sufficient.",
            decided_by_agent_id=decision.authority_agent_id,
        )
    result = await service.tick(db_session, run.id)

    await db_session.refresh(goal)
    await db_session.refresh(run)
    await db_session.refresh(gate)
    assert result["run_completed"] is True, result
    events = await _events(db_session, test_project.id, "orchestration.run_completed")
    payload = events[0].payload
    assert goal.status == "completed"
    assert run.status == "completed"
    assert gate.status == "accepted"
    assert {item["key"] for item in payload["declared_success_criteria"]} == {"expanded", "validated"}
    assert {item["criterion_key"] for item in payload["criterion_evidence"]} == {
        "expanded",
        "validated",
    }
    assert {item["success_criterion_key"] for item in payload["accepted_non_summary_gates"]} == {
        "plan",
        "plan_item:implement-api",
    }


@pytest.mark.asyncio
async def test_explicit_exceeded_budget_blocks_completion(db_session, test_project):
    service, goal, run, writer, task, gate = await _request_summary(db_session, test_project)
    run.budget_state = {"status": "exceeded", "overridden": False}
    session = Session(
        agent_id=writer.id,
        task_id=task.id,
        project_id=test_project.id,
        adapter_type="api",
        status="completed",
        output=await _valid_summary_output(db_session, goal, run),
        metadata_={},
        origin="auto",
    )
    db_session.add(session)
    await db_session.flush()
    await emit_event_once(
        db_session,
        test_project.id,
        "session.completed",
        {"session_id": str(session.id), "task_id": str(task.id)},
        dedup_key=f"phase17-session-completed:{session.id}",
    )

    result = await service.tick(db_session, run.id)

    await db_session.refresh(goal)
    await db_session.refresh(run)
    await db_session.refresh(gate)
    assert gate.status == "accepted"
    assert result["run_completed"] is False
    assert goal.status == "active"
    assert run.status == "running"


@pytest.mark.asyncio
async def test_tick_resumes_reserved_complete_run_action(db_session, test_project):
    service, goal, run, writer, task, _gate = await _request_summary(db_session, test_project)
    goal.weight = "trivial"
    await heal_baseline_drift_for_test(db_session, goal, run)
    session = Session(
        agent_id=writer.id,
        task_id=task.id,
        project_id=test_project.id,
        adapter_type="api",
        status="completed",
        output=await _valid_summary_output(db_session, goal, run),
        metadata_={},
        origin="auto",
    )
    db_session.add(session)
    await db_session.flush()
    await emit_event_once(
        db_session,
        test_project.id,
        "session.completed",
        {"session_id": str(session.id), "task_id": str(task.id)},
        dedup_key=f"phase17-session-completed:{session.id}",
    )
    reserved = await service.reserve_action(
        db_session,
        run.id,
        f"run:{run.id}:kind:complete_run",
        "complete_run",
        {
            "action_type": "complete_run",
            "reason": "All gates and final summary evidence are accepted.",
        },
    )

    result = await service.tick(db_session, run.id)

    await db_session.refresh(goal)
    await db_session.refresh(reserved)
    assert result["completion_action_id"] == reserved.id
    assert reserved.status == "completed"
    assert goal.status == "completed"


@pytest.mark.asyncio
async def test_invalid_summary_creates_one_replacement_and_valid_replacement_completes(
    db_session,
    test_project,
):
    service, goal, run, writer, task, gate = await _request_summary(db_session, test_project)
    goal.weight = "trivial"
    await heal_baseline_drift_for_test(db_session, goal, run)
    request_action_id = task.metadata_["orchestration_final_summary"]["request_action_id"]
    accepted_gate_ids = task.metadata_["orchestration_final_summary"]["accepted_gate_ids"]
    accepted_evidence_ids = task.metadata_["orchestration_final_summary"]["accepted_evidence_ids"]
    invalid_session = Session(
        agent_id=writer.id,
        task_id=task.id,
        project_id=test_project.id,
        adapter_type="api",
        status="completed",
        output="not json",
        metadata_={},
        origin="auto",
    )
    db_session.add(invalid_session)
    await db_session.flush()
    await emit_event_once(
        db_session,
        test_project.id,
        "session.completed",
        {"session_id": str(invalid_session.id), "task_id": str(task.id)},
        dedup_key=f"phase17-invalid-replacement:{invalid_session.id}",
    )

    failed_tick = await service.tick(db_session, run.id)

    failed_evidence = (
        await db_session.execute(
            select(OrchestrationEvidence).where(
                OrchestrationEvidence.gate_id == gate.id,
                OrchestrationEvidence.source_id == invalid_session.id,
            )
        )
    ).scalar_one()
    replacement_actions = list(
        (
            await db_session.execute(
                select(OrchestrationAction).where(
                    OrchestrationAction.run_id == run.id,
                    OrchestrationAction.idempotency_key
                    == f"run:{run.id}:kind:create_delegation_task:final_summary:replacement:{failed_evidence.id}",
                )
            )
        ).scalars().all()
    )
    assert failed_tick["run_completed"] is False
    assert failed_tick["recoveries_created"] == 1
    assert len(replacement_actions) == 1
    replacement_action = replacement_actions[0]
    replacement_task = await db_session.get(Task, replacement_action.target_id)
    await db_session.refresh(gate)
    assert gate.status == "failed"
    assert replacement_action.action_type == "create_delegation_task"
    assert replacement_task.id != task.id
    assert replacement_task.metadata_["orchestration"]["gate_id"] == str(gate.id)
    assert replacement_task.metadata_["orchestration"]["final_summary"] is True
    assert replacement_task.metadata_["orchestration_final_summary"] == {
        "request_action_id": request_action_id,
        "gate_id": str(gate.id),
        "accepted_gate_ids": accepted_gate_ids,
        "accepted_evidence_ids": accepted_evidence_ids,
    }
    await _complete_goal_definition(db_session, goal, run)

    replay = await service.tick(db_session, run.id)
    assert replay["recoveries_created"] == 0
    assert await db_session.scalar(
        select(func.count(OrchestrationAction.id)).where(
            OrchestrationAction.run_id == run.id,
            OrchestrationAction.idempotency_key
            == f"run:{run.id}:kind:create_delegation_task:final_summary:replacement:{failed_evidence.id}",
        )
    ) == 1

    valid_session = Session(
        agent_id=replacement_task.assigned_to,
        task_id=replacement_task.id,
        project_id=test_project.id,
        adapter_type="api",
        status="completed",
        output=await _valid_summary_output(db_session, goal, run),
        metadata_={},
        origin="auto",
    )
    db_session.add(valid_session)
    await db_session.flush()
    await emit_event_once(
        db_session,
        test_project.id,
        "session.completed",
        {"session_id": str(valid_session.id), "task_id": str(replacement_task.id)},
        dedup_key=f"phase17-valid-replacement:{valid_session.id}",
    )

    completed_tick = await service.tick(db_session, run.id)

    replacement_evidence = (
        await db_session.execute(
            select(OrchestrationEvidence).where(
                OrchestrationEvidence.gate_id == gate.id,
                OrchestrationEvidence.source_id == valid_session.id,
            )
        )
    ).scalar_one()
    completed_event = (await _events(db_session, test_project.id, "orchestration.run_completed"))[0]
    await db_session.refresh(gate)
    assert completed_tick["run_completed"] is True
    assert gate.status == "accepted"
    assert completed_event.payload["final_summary"] == {
        "gate_id": str(gate.id),
        "evidence_id": str(replacement_evidence.id),
        "task_id": str(replacement_task.id),
        "session_id": str(valid_session.id),
        "agent_id": str(replacement_task.assigned_to),
    }


@pytest.mark.asyncio
async def test_replacement_summary_key_is_stable_per_failed_evidence(db_session, test_project):
    service, _goal, run, writer, task, gate = await _request_summary(db_session, test_project)
    first_session = Session(
        agent_id=writer.id,
        task_id=task.id,
        project_id=test_project.id,
        adapter_type="api",
        status="completed",
        output="bad one",
        metadata_={},
        origin="auto",
    )
    db_session.add(first_session)
    await db_session.flush()
    await emit_event_once(
        db_session,
        test_project.id,
        "session.completed",
        {"session_id": str(first_session.id), "task_id": str(task.id)},
        dedup_key=f"phase17-stable-replacement:{first_session.id}",
    )
    await service.tick(db_session, run.id)
    first_evidence = (
        await db_session.execute(
            select(OrchestrationEvidence).where(
                OrchestrationEvidence.gate_id == gate.id,
                OrchestrationEvidence.source_id == first_session.id,
            )
        )
    ).scalar_one()
    first_key = f"run:{run.id}:kind:create_delegation_task:final_summary:replacement:{first_evidence.id}"
    first_action = (
        await db_session.execute(
            select(OrchestrationAction).where(OrchestrationAction.idempotency_key == first_key)
        )
    ).scalar_one()
    replacement_task = await db_session.get(Task, first_action.target_id)
    await _complete_goal_definition(db_session, _goal, run)

    await service.tick(db_session, run.id)
    assert await db_session.scalar(
        select(func.count(OrchestrationAction.id)).where(
            OrchestrationAction.idempotency_key == first_key
        )
    ) == 1

    second_session = Session(
        agent_id=replacement_task.assigned_to,
        task_id=replacement_task.id,
        project_id=test_project.id,
        adapter_type="api",
        status="completed",
        output="bad two",
        metadata_={},
        origin="auto",
    )
    db_session.add(second_session)
    await db_session.flush()
    await emit_event_once(
        db_session,
        test_project.id,
        "session.completed",
        {"session_id": str(second_session.id), "task_id": str(replacement_task.id)},
        dedup_key=f"phase17-stable-replacement:{second_session.id}",
    )
    await service.tick(db_session, run.id)
    second_evidence = (
        await db_session.execute(
            select(OrchestrationEvidence).where(
                OrchestrationEvidence.gate_id == gate.id,
                OrchestrationEvidence.source_id == second_session.id,
            )
        )
    ).scalar_one()
    second_key = f"run:{run.id}:kind:create_delegation_task:final_summary:replacement:{second_evidence.id}"
    assert second_key != first_key
    assert await db_session.scalar(
        select(func.count(OrchestrationAction.id)).where(
            OrchestrationAction.idempotency_key.in_([first_key, second_key])
        )
    ) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_status", ["paused", "cancelled"])
async def test_terminal_status_after_recovery_blocks_completion(
    db_session,
    test_project,
    monkeypatch,
    terminal_status,
):
    service, goal, run, writer, task, _gate = await _request_summary(db_session, test_project)
    session = Session(
        agent_id=writer.id,
        task_id=task.id,
        project_id=test_project.id,
        adapter_type="api",
        status="completed",
        output=await _valid_summary_output(db_session, goal, run),
        metadata_={},
        origin="auto",
    )
    db_session.add(session)
    await db_session.flush()
    await emit_event_once(
        db_session,
        test_project.id,
        "session.completed",
        {"session_id": str(session.id), "task_id": str(task.id)},
        dedup_key=f"phase17-terminal-after-recovery:{terminal_status}:{session.id}",
    )

    async def terminal_recovery(db, run_id, baseline_ready=False):
        await db.execute(
            update(OrchestrationRun)
            .where(OrchestrationRun.id == run_id)
            .values(status=terminal_status)
        )
        return 0

    monkeypatch.setattr(service, "recover_run", terminal_recovery)

    result = await service.tick(db_session, run.id)

    await db_session.refresh(run)
    await db_session.refresh(goal)
    assert result["run_completed"] is False
    assert run.status == terminal_status
    assert goal.status == "active"
    assert await db_session.scalar(
        select(func.count(OrchestrationAction.id)).where(
            OrchestrationAction.run_id == run.id,
            OrchestrationAction.action_type == "complete_run",
        )
    ) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_run_rechecks_terminal_status_before_reservation(
    db_session,
    test_project,
    monkeypatch,
):
    writer = _agent("writer", "technical writer", ["summarization"])
    db_session.add(writer)
    await db_session.flush()
    service, goal, run = await _make_run(db_session, test_project.id)
    await _accepted_gate(db_session, run.id, "implemented")
    await _accepted_gate(db_session, run.id, "validated")
    request_action = await service.execute_request_final_summary_action(
        db_session,
        run.id,
        {"action_type": "request_final_summary", "work_function": "summarization"},
        f"run:{run.id}:kind:request_final_summary",
    )
    task = await db_session.get(Task, request_action.target_id)
    session = Session(
        agent_id=writer.id,
        task_id=task.id,
        project_id=test_project.id,
        adapter_type="api",
        status="completed",
        output=await _valid_summary_output(db_session, goal, run),
        metadata_={},
        origin="auto",
    )
    db_session.add(session)
    await db_session.flush()
    event, _ = await emit_event_once(
        db_session,
        test_project.id,
        "session.completed",
        {"session_id": str(session.id), "task_id": str(task.id)},
        dedup_key=f"phase17-recheck-before-reservation:{session.id}",
    )
    event = (await db_session.execute(select(EventLog).where(EventLog.id == event.id))).scalar_one()
    assert await service._ingest_evidence_from_event(db_session, run, event) == 1
    assert await service.validate_open_gates(db_session, run.id) == 1
    await _authorize_completion_closeout(db_session, service, goal, run)
    original_manifest = service._completion_manifest

    async def terminal_manifest(db, goal_to_complete, run_to_complete):
        manifest = await original_manifest(db, goal_to_complete, run_to_complete)
        goal_to_complete.status = "cancelled"
        await db.flush()
        return manifest

    monkeypatch.setattr(service, "_completion_manifest", terminal_manifest)

    with pytest.raises(HTTPException) as exc:
        await service.execute_complete_run_action(
            db_session,
            run.id,
            {
                "action_type": "complete_run",
                "reason": "All gates and final summary evidence are accepted.",
            },
            f"run:{run.id}:kind:complete_run",
        )

    assert exc.value.status_code == 409
    assert await db_session.scalar(
        select(func.count(OrchestrationAction.id)).where(
            OrchestrationAction.run_id == run.id,
            OrchestrationAction.action_type == "complete_run",
        )
    ) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_status", ["paused", "cancelled"])
async def test_recover_run_skips_final_summary_replacement_for_terminal_goal(
    db_session,
    test_project,
    terminal_status,
):
    service, goal, run, writer, task, gate = await _request_summary(db_session, test_project)
    session = Session(
        agent_id=writer.id,
        task_id=task.id,
        project_id=test_project.id,
        adapter_type="api",
        status="completed",
        output="not json",
        metadata_={},
        origin="auto",
    )
    db_session.add(session)
    await db_session.flush()
    event, _ = await emit_event_once(
        db_session,
        test_project.id,
        "session.completed",
        {"session_id": str(session.id), "task_id": str(task.id)},
        dedup_key=f"phase17-terminal-goal-replacement:{terminal_status}:{session.id}",
    )
    event = (await db_session.execute(select(EventLog).where(EventLog.id == event.id))).scalar_one()
    assert await service._ingest_evidence_from_event(db_session, run, event) == 1
    assert await service.validate_open_gates(db_session, run.id) == 1
    await db_session.refresh(gate)
    failed_evidence = (
        await db_session.execute(
            select(OrchestrationEvidence).where(
                OrchestrationEvidence.gate_id == gate.id,
                OrchestrationEvidence.source_id == session.id,
            )
        )
    ).scalar_one()
    goal.status = terminal_status
    await db_session.flush()

    created = await service.recover_run(db_session, run.id)

    await db_session.refresh(run)
    assert created == 0
    assert run.status == "running"
    assert await db_session.scalar(
        select(func.count(OrchestrationAction.id)).where(
            OrchestrationAction.idempotency_key
            == f"run:{run.id}:kind:create_delegation_task:final_summary:replacement:{failed_evidence.id}"
        )
    ) == 0
    assert await db_session.scalar(
        select(func.count(Task.id)).where(
            Task.metadata_["orchestration"]["final_summary"].as_boolean().is_(True)
        )
    ) == 1


@pytest.mark.asyncio
async def test_failed_complete_run_action_does_not_report_run_completed(db_session, test_project):
    service, goal, run, writer, task, _gate = await _request_summary(db_session, test_project)
    goal.weight = "trivial"
    await heal_baseline_drift_for_test(db_session, goal, run)
    session = Session(
        agent_id=writer.id,
        task_id=task.id,
        project_id=test_project.id,
        adapter_type="api",
        status="completed",
        output=await _valid_summary_output(db_session, goal, run),
        metadata_={},
        origin="auto",
    )
    db_session.add(session)
    await db_session.flush()
    await emit_event_once(
        db_session,
        test_project.id,
        "session.completed",
        {"session_id": str(session.id), "task_id": str(task.id)},
        dedup_key=f"phase17-failed-completion:{session.id}",
    )
    failed_action = await service.reserve_action(
        db_session,
        run.id,
        f"run:{run.id}:kind:complete_run",
        "complete_run",
        {
            "action_type": "complete_run",
            "reason": "All gates and final summary evidence are accepted.",
        },
    )
    await service.mark_action_failed(db_session, failed_action, "simulated completion failure")

    result = await service.tick(db_session, run.id)

    await db_session.refresh(failed_action)
    await db_session.refresh(goal)
    await db_session.refresh(run)
    assert result["run_completed"] is False
    assert result["completion_action_id"] == failed_action.id
    assert failed_action.status == "failed"
    assert goal.status == "active"
    assert run.status == "running"
    assert await _events(db_session, test_project.id, "orchestration.run_completed") == []
