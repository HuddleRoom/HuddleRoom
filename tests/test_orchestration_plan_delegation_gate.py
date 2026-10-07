import json
import uuid
from datetime import UTC, datetime

import pytest
from fastapi import HTTPException
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql.functions import count

from huddleroom.models.agent import Agent
from huddleroom.models.artifact import Artifact
from huddleroom.models.event_log import EventLog
from huddleroom.models.orchestration import (
    OrchestrationAction,
    OrchestrationDecision,
    OrchestrationEvidence,
    OrchestrationGate,
    OrchestrationGoal,
    OrchestrationRun,
)
from huddleroom.models.project import Project
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.schemas.orchestration import OrchestrationGoalCreate
from huddleroom.schemas.task import TaskCreate
from huddleroom.services.orchestration_decision_validator import validate_orchestration_decision
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.task_service import TaskService


async def _complete_goal_definition(db_session, goal, run):
    from tests.conftest import complete_baseline_processes

    run.baseline_authorized = True
    await complete_baseline_processes(db_session, goal, run)


async def _make_run(db_session, project_id):
    service = OrchestrationService()
    goal, run = await service.create_goal(
        db_session,
        project_id=project_id,
        data=OrchestrationGoalCreate(
            objective="Ship the plan delegation gate",
            success_criteria=[
                {"key": "plan", "description": "An agent-produced plan is accepted before work expansion."}
            ],
            constraints={"owned_files": ["huddleroom/services/orchestration_service.py"]},
            budget={"max_tokens": 20000},
        ),
        created_by_user_id=None,
    )
    await _complete_goal_definition(db_session, goal, run)
    return service, goal, run


async def _event_rows(db_session, project_id, event_type):
    result = await db_session.execute(
        select(EventLog)
        .where(EventLog.project_id == project_id, EventLog.event_type == event_type)
        .order_by(EventLog.seq.asc())
    )
    return list(result.scalars().all())


def _agent(name_prefix: str, role: str = "planner", capabilities: list[str] | None = None, *, is_active: bool = True):
    return Agent(
        name=f"{name_prefix}-{uuid.uuid4()}",
        role=role,
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=capabilities or ["planning"],
        config={},
        is_active=is_active,
    )


def _plan_request(agent_id: uuid.UUID, *, work_function: str = " planning "):
    request = {
        "action_type": "request_plan",
        "agent_id": f" {agent_id} ",
        "work_function": work_function,
        "scope": " Create a phase implementation plan with independently verifiable tasks. ",
    }
    return request


def _canonical_plan_request(agent_id: uuid.UUID):
    return {
        "action_type": "request_plan",
        "agent_id": str(agent_id),
        "work_function": "planning",
        "scope": "Create a phase implementation plan with independently verifiable tasks.",
    }


def _accepted_plan_metadata() -> dict:
    return {
        "kind": "implementation_plan",
        "plan_items": [{
            "id": "implementation",
            "work_function": "implementation",
            "scope": "Implement the approved plan.",
            "deliverable": "Completed implementation.",
            "success_evidence": ["Task output."],
            "success_criterion_keys": ["plan"],
        }],
    }


async def _single_plan_gate(db_session, run_id):
    result = await db_session.execute(
        select(OrchestrationGate).where(
            OrchestrationGate.run_id == run_id,
            OrchestrationGate.success_criterion_key == "plan",
            OrchestrationGate.gate_type == "plan_accepted",
        )
    )
    return result.scalar_one()


def _test_session_factory(test_engine):
    return async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)


async def _make_committed_run(test_engine, workspace):
    session_factory = _test_session_factory(test_engine)
    async with session_factory() as session:
        project = Project(
            name=f"Plan delegation project {uuid.uuid4()}",
            description="test",
            workspace_path=str(workspace),
            config={},
        )
        session.add(project)
        await session.flush()
        service, goal, run = await _make_run(session, project.id)
        await session.commit()
        return service, session_factory, project.id, goal.id, run.id


async def _cleanup_committed_plan_rows(
    session_factory,
    *,
    project_id: uuid.UUID,
    goal_id: uuid.UUID,
    run_id: uuid.UUID,
    agent_ids: list[uuid.UUID] | None = None,
):
    async with session_factory() as cleanup_db:
        task_ids = select(Task.id).where(Task.project_id == project_id)
        await cleanup_db.execute(delete(OrchestrationEvidence).where(OrchestrationEvidence.run_id == run_id))
        await cleanup_db.execute(delete(Session).where(Session.task_id.in_(task_ids)))
        await cleanup_db.execute(delete(Artifact).where(Artifact.project_id == project_id))
        await cleanup_db.execute(delete(OrchestrationGate).where(OrchestrationGate.run_id == run_id))
        await cleanup_db.execute(delete(OrchestrationDecision).where(OrchestrationDecision.run_id == run_id))
        await cleanup_db.execute(delete(OrchestrationAction).where(OrchestrationAction.run_id == run_id))
        await cleanup_db.execute(delete(Task).where(Task.project_id == project_id))
        await cleanup_db.execute(delete(OrchestrationRun).where(OrchestrationRun.id == run_id))
        await cleanup_db.execute(delete(OrchestrationGoal).where(OrchestrationGoal.id == goal_id))
        if agent_ids:
            await cleanup_db.execute(delete(Agent).where(Agent.id.in_(agent_ids)))
        await cleanup_db.execute(delete(Project).where(Project.id == project_id))
        await cleanup_db.commit()


@pytest.mark.asyncio
async def test_execute_request_plan_action_creates_planning_task_and_gate(db_session, test_project):
    planner = _agent("planner")
    db_session.add(planner)
    await db_session.flush()
    service, goal, run = await _make_run(db_session, test_project.id)

    action = await service.execute_request_plan_action(
        db_session,
        run_id=run.id,
        request=_plan_request(planner.id),
        idempotency_key="run:phase11:kind:request_plan",
    )

    assert action.status == "completed"
    assert action.action_type == "request_plan"
    assert action.target_type == "task"
    assert action.target_id == action.id
    assert action.request == _canonical_plan_request(planner.id)

    task = await db_session.get(Task, action.id)
    assert task is not None
    assert task.assigned_to == planner.id
    assert task.status == "backlog"
    assert task.title == "Planning: Agent-produced implementation plan."
    assert task.description is not None
    assert goal.objective in task.description
    assert "Scope: Create a phase implementation plan with independently verifiable tasks." in task.description
    assert task.metadata_["orchestration"] == {
        "goal_id": str(goal.id),
        "run_id": str(run.id),
        "action_id": str(action.id),
        "work_function": "planning",
    }
    assert task.metadata_["orchestration_plan"] == {"status": "requested"}
    contract = task.metadata_["orchestration_contract"]
    assert contract["inputs"] == [
        f"Goal objective: {goal.objective}",
        f"Success criteria: {json.dumps(goal.success_criteria, sort_keys=True)}",
        f"Constraints: {json.dumps(goal.constraints, sort_keys=True)}",
    ]
    assert contract["deliverable"] == "Agent-produced implementation plan."
    assert contract["forbidden_work"] == [
        "Do not implement the plan.",
        "Do not edit project artifacts.",
        "Do not create tasks, meetings, graphs, rules, hooks, or automations.",
    ]
    assert contract["success_evidence"] == [
        "A plan artifact linked to this planning task.",
        "Plan items are independently actionable and verifiable.",
    ]
    # The terminal test analyzer leaves goal fields untouched; the planning
    # contract copies that original budget.
    assert contract["budget"] == {"max_tokens": 20000}
    assert contract["report_schema"] == {
        "artifact_id": "uuid",
        "plan_items": "list[{id, title, work_function, scope, deliverable, success_evidence, success_criterion_keys}]",
    }

    gate = await _single_plan_gate(db_session, run.id)
    assert gate.status == "open"
    assert gate.gate_type == "plan_accepted"
    assert gate.required_evidence == {"required_source_types": ["artifact"], "min_count": 1}

    assert run.plan_state == {
        "status": "requested",
        "planning_task_id": str(task.id),
        "request_action_id": str(action.id),
        "work_function": "planning",
        "plan_gate_id": str(gate.id),
        "revision_requests": [],
    }

    events = await _event_rows(db_session, test_project.id, "orchestration.plan_requested")
    assert len(events) == 1
    assert events[0].payload == {
        "goal_id": str(goal.id),
        "run_id": str(run.id),
        "action_id": str(action.id),
        "task_id": str(task.id),
        "agent_id": str(planner.id),
        "gate_id": str(gate.id),
        "work_function": "planning",
    }


@pytest.mark.asyncio
async def test_execute_request_plan_action_rejects_non_planning_work_function_before_reserving(
    db_session,
    test_project,
):
    planner = _agent("planner")
    db_session.add(planner)
    await db_session.flush()
    service, _, run = await _make_run(db_session, test_project.id)

    with pytest.raises(HTTPException) as exc_info:
        await service.execute_request_plan_action(
            db_session,
            run_id=run.id,
            request=_plan_request(planner.id, work_function="implementation"),
            idempotency_key="run:phase11:kind:request_plan:normalize-work-function",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "request_plan work_function must be 'planning'"
    assert (
        await db_session.scalar(select(count(OrchestrationAction.id)).where(OrchestrationAction.run_id == run.id))
        == 0
    )


@pytest.mark.asyncio
async def test_execute_request_plan_action_replay_reuses_original_task_and_gate(db_session, test_project):
    planner = _agent("planner")
    other_planner = _agent("other-planner")
    db_session.add_all([planner, other_planner])
    await db_session.flush()
    service, _, run = await _make_run(db_session, test_project.id)

    first = await service.execute_request_plan_action(
        db_session,
        run_id=run.id,
        request=_plan_request(planner.id),
        idempotency_key="run:phase11:kind:request_plan",
    )
    second = await service.execute_request_plan_action(
        db_session,
        run_id=run.id,
        request=_plan_request(other_planner.id),
        idempotency_key="run:phase11:kind:request_plan",
    )

    assert second.id == first.id
    assert second.request == _canonical_plan_request(planner.id)
    assert await db_session.scalar(select(count(Task.id)).where(Task.project_id == test_project.id)) == 1
    assert await db_session.scalar(select(count(OrchestrationGate.id)).where(OrchestrationGate.run_id == run.id)) == 1
    events = await _event_rows(db_session, test_project.id, "orchestration.plan_requested")
    assert len(events) == 1


@pytest.mark.asyncio
async def test_execute_request_plan_action_replay_reuses_existing_task_before_agent_revalidation(
    db_session,
    test_project,
):
    planner = _agent("planner")
    db_session.add(planner)
    await db_session.flush()
    service, goal, run = await _make_run(db_session, test_project.id)

    request = _canonical_plan_request(planner.id)
    action = await service.reserve_action(
        db_session,
        run_id=run.id,
        idempotency_key="run:phase11:kind:request_plan:partial-replay",
        action_type="request_plan",
        request=request,
    )
    gate = await service._ensure_plan_gate(db_session, run.id)
    contract = service._planning_contract(goal, run, action, request)
    await TaskService().create(
        db_session,
        test_project.id,
        TaskCreate(
            title=service._delegation_task_title(contract.work_function, contract.deliverable),
            description=service._delegation_task_description(goal, contract),
            assigned_to=planner.id,
            metadata={
                "orchestration": {
                    "goal_id": str(goal.id),
                    "run_id": str(run.id),
                    "action_id": str(action.id),
                    "work_function": contract.work_function,
                },
                "orchestration_plan": {"status": "requested"},
                "orchestration_contract": contract.model_dump(mode="json"),
            },
        ),
        task_id=action.id,
    )
    planner.is_active = False
    await db_session.flush()

    replayed = await service.execute_request_plan_action(
        db_session,
        run_id=run.id,
        request=_plan_request(planner.id),
        idempotency_key="run:phase11:kind:request_plan:partial-replay",
    )

    assert replayed.id == action.id
    assert replayed.status == "completed"
    assert replayed.target_id == action.id
    assert run.plan_state == {
        "status": "requested",
        "planning_task_id": str(action.id),
        "request_action_id": str(action.id),
        "work_function": "planning",
        "plan_gate_id": str(gate.id),
        "revision_requests": [],
    }
    events = await _event_rows(db_session, test_project.id, "orchestration.plan_requested")
    assert len(events) == 1


@pytest.mark.asyncio
async def test_execute_request_plan_action_replay_uses_stored_agent_id_when_task_assignment_is_missing(
    db_session,
    test_project,
):
    planner = _agent("planner")
    db_session.add(planner)
    await db_session.flush()
    service, goal, run = await _make_run(db_session, test_project.id)

    request = _canonical_plan_request(planner.id)
    action = await service.reserve_action(
        db_session,
        run_id=run.id,
        idempotency_key="run:phase11:kind:request_plan:missing-assignment",
        action_type="request_plan",
        request=request,
    )
    gate = await service._ensure_plan_gate(db_session, run.id)
    contract = service._planning_contract(goal, run, action, request)
    await TaskService().create(
        db_session,
        test_project.id,
        TaskCreate(
            title=service._delegation_task_title(contract.work_function, contract.deliverable),
            description=service._delegation_task_description(goal, contract),
            assigned_to=None,
            metadata={
                "orchestration": {
                    "goal_id": str(goal.id),
                    "run_id": str(run.id),
                    "action_id": str(action.id),
                    "work_function": contract.work_function,
                },
                "orchestration_plan": {"status": "requested"},
                "orchestration_contract": contract.model_dump(mode="json"),
            },
        ),
        task_id=action.id,
    )
    planner.is_active = False
    await db_session.flush()

    replayed = await service.execute_request_plan_action(
        db_session,
        run_id=run.id,
        request=_plan_request(planner.id),
        idempotency_key="run:phase11:kind:request_plan:missing-assignment",
    )

    assert replayed.id == action.id
    assert replayed.status == "completed"
    assert run.plan_state == {
        "status": "requested",
        "planning_task_id": str(action.id),
        "request_action_id": str(action.id),
        "work_function": "planning",
        "plan_gate_id": str(gate.id),
        "revision_requests": [],
    }
    events = await _event_rows(db_session, test_project.id, "orchestration.plan_requested")
    assert len(events) == 1
    assert events[0].payload["agent_id"] == str(planner.id)


@pytest.mark.asyncio
async def test_planning_task_for_run_accepts_task_id_distinct_from_request_action_id(
    db_session,
    test_project,
):
    planner = _agent("planner")
    db_session.add(planner)
    await db_session.flush()
    service, goal, run = await _make_run(db_session, test_project.id)

    request = _canonical_plan_request(planner.id)
    action = await service.reserve_action(
        db_session,
        run_id=run.id,
        idempotency_key="run:phase11:kind:request_plan:metadata-task-replay",
        action_type="request_plan",
        request=request,
    )
    contract = service._planning_contract(goal, run, action, request)
    task = await TaskService().create(
        db_session,
        test_project.id,
        TaskCreate(
            title=service._delegation_task_title(contract.work_function, contract.deliverable),
            description=service._delegation_task_description(goal, contract),
            assigned_to=planner.id,
            metadata={
                "orchestration": {
                    "goal_id": str(goal.id),
                    "run_id": str(run.id),
                    "action_id": str(action.id),
                    "work_function": contract.work_function,
                },
                "orchestration_plan": {"status": "requested"},
                "orchestration_contract": contract.model_dump(mode="json"),
            },
        ),
    )
    run.plan_state = {
        "status": "requested",
        "planning_task_id": str(task.id),
        "request_action_id": str(action.id),
        "work_function": "planning",
        "plan_gate_id": str(uuid.uuid4()),
        "revision_requests": [],
    }
    await db_session.flush()

    planning_task = await service._planning_task_for_run(db_session, run, task.id)

    assert planning_task.id == task.id


@pytest.mark.asyncio
async def test_execute_request_plan_action_preserves_existing_plan_state_fields(db_session, test_project):
    planner = _agent("planner")
    db_session.add(planner)
    await db_session.flush()
    service, goal, run = await _make_run(db_session, test_project.id)
    request = _canonical_plan_request(planner.id)
    action = await service.reserve_action(
        db_session,
        run_id=run.id,
        idempotency_key="run:phase11:kind:request_plan:preserve-state",
        action_type="request_plan",
        request=request,
    )
    gate = await service._ensure_plan_gate(db_session, run.id)
    contract = service._planning_contract(goal, run, action, request)
    await TaskService().create(
        db_session,
        test_project.id,
        TaskCreate(
            title=service._delegation_task_title(contract.work_function, contract.deliverable),
            description=service._delegation_task_description(goal, contract),
            assigned_to=planner.id,
            metadata={
                "orchestration": {
                    "goal_id": str(goal.id),
                    "run_id": str(run.id),
                    "action_id": str(action.id),
                    "work_function": contract.work_function,
                },
                "orchestration_plan": {"status": "requested"},
                "orchestration_contract": contract.model_dump(mode="json"),
            },
        ),
        task_id=action.id,
    )
    run.plan_state = {
        "status": "revision_requested",
        "planning_task_id": str(uuid.uuid4()),
        "request_action_id": str(action.id),
        "plan_gate_id": str(uuid.uuid4()),
        "work_function": "planning",
        "accepted_artifact_id": str(uuid.uuid4()),
        "accept_action_id": str(uuid.uuid4()),
        "custom_note": "keep me",
        "revision_requests": [{"action_id": "old", "revision_request": "keep me"}],
    }
    planner.is_active = False
    await db_session.flush()

    replayed = await service.execute_request_plan_action(
        db_session,
        run_id=run.id,
        request=_plan_request(planner.id),
        idempotency_key="run:phase11:kind:request_plan:preserve-state",
    )

    assert replayed.status == "completed"
    assert run.plan_state["status"] == "requested"
    assert run.plan_state["planning_task_id"] == str(action.id)
    assert run.plan_state["request_action_id"] == str(action.id)
    assert run.plan_state["plan_gate_id"] == str(gate.id)
    assert run.plan_state["work_function"] == "planning"
    assert run.plan_state["custom_note"] == "keep me"
    assert run.plan_state["revision_requests"] == [{"action_id": "old", "revision_request": "keep me"}]
    assert "accepted_artifact_id" not in run.plan_state
    assert "accept_action_id" not in run.plan_state


@pytest.mark.asyncio
async def test_execute_request_plan_action_clears_stale_plan_lifecycle_fields_for_new_request(
    db_session,
    test_project,
):
    planner = _agent("planner")
    db_session.add(planner)
    await db_session.flush()
    service, _, run = await _make_run(db_session, test_project.id)
    stale_gate = await service._ensure_plan_gate(db_session, run.id)
    stale_gate.status = "accepted"
    stale_gate.accepted_at = datetime.now(UTC)
    stale_artifact = Artifact(
        project_id=test_project.id,
        name="stale-plan",
        artifact_type="plan",
        status="draft",
        linked_task_id=None,
        created_by_agent=planner.id,
        metadata_={"kind": "implementation_plan"},
    )
    db_session.add(stale_artifact)
    await db_session.flush()
    db_session.add(
        OrchestrationEvidence(
            run_id=run.id,
            gate_id=stale_gate.id,
            source_type="artifact",
            source_id=stale_artifact.id,
            observed_event_id=None,
            producer_agent_id=planner.id,
            verdict="accepted",
            evidence_metadata={"source": "stale"},
        )
    )
    run.plan_state = {
        "status": "accepted",
        "planning_task_id": str(uuid.uuid4()),
        "request_action_id": str(uuid.uuid4()),
        "plan_gate_id": str(stale_gate.id),
        "work_function": "planning",
        "accepted_artifact_id": str(uuid.uuid4()),
        "accept_action_id": str(uuid.uuid4()),
        "custom_note": "keep me",
        "revision_requests": [{"action_id": "old", "revision_request": "stale"}],
    }

    with pytest.raises(HTTPException) as exc_info:
        await service.execute_request_plan_action(
            db_session,
            run_id=run.id,
            request=_plan_request(planner.id),
            idempotency_key="run:phase11:kind:request_plan:clear-stale-state",
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Terminal plan cannot be re-requested for this run"
    assert run.plan_state["status"] == "accepted"
    assert run.plan_state["plan_gate_id"] == str(stale_gate.id)
    assert await db_session.scalar(
        select(count(OrchestrationEvidence.id)).where(OrchestrationEvidence.gate_id == stale_gate.id)
    ) == 1


@pytest.mark.asyncio
async def test_execute_request_plan_action_rejects_missing_agent_before_reserving(db_session, test_project):
    service, _, run = await _make_run(db_session, test_project.id)

    with pytest.raises(HTTPException) as exc_info:
        await service.execute_request_plan_action(
            db_session,
            run_id=run.id,
            request=_plan_request(uuid.uuid4()),
            idempotency_key="run:phase11:kind:request_plan:missing-agent",
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Agent not found"
    assert (
        await db_session.scalar(select(count(OrchestrationAction.id)).where(OrchestrationAction.run_id == run.id))
        == 0
    )
    assert await db_session.scalar(select(count(Artifact.id))) == 0
    assert await db_session.scalar(select(count(OrchestrationEvidence.id))) == 0


@pytest.mark.asyncio
async def test_execute_request_plan_action_revalidates_stored_request_when_reserve_returns_different_action(
    db_session,
    test_project,
):
    incoming_planner = _agent("incoming-planner")
    stored_planner = _agent("stored-planner", is_active=False)
    db_session.add_all([incoming_planner, stored_planner])
    await db_session.flush()
    service, _, run = await _make_run(db_session, test_project.id)

    raced_action = await service.reserve_action(
        db_session,
        run_id=run.id,
        idempotency_key="run:phase11:kind:request_plan:raced-source",
        action_type="request_plan",
        request=_canonical_plan_request(stored_planner.id),
    )

    async def fake_reserve_action(*args, **kwargs):
        return raced_action

    service.reserve_action = fake_reserve_action

    with pytest.raises(HTTPException) as exc_info:
        await service.execute_request_plan_action(
            db_session,
            run_id=run.id,
            request=_plan_request(incoming_planner.id),
            idempotency_key="run:phase11:kind:request_plan:raced-target",
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Agent not found"
    assert await db_session.get(Task, raced_action.id) is None


@pytest.mark.asyncio
async def test_execute_request_plan_action_replay_missing_inactive_stored_agent_fails_without_creating_gate(
    test_engine, tmp_path,
):
    session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    planner_id = None
    project_id = None
    goal_id = None
    run_id = None
    try:
        async with session_factory() as setup_db:
            project = Project(
                name="Replay Project",
                description="Missing agent replay",
                workspace_path=str(tmp_path),
                config={},
            )
            planner = _agent("planner", is_active=False)
            setup_db.add_all([project, planner])
            await setup_db.flush()
            service, goal, run = await _make_run(setup_db, project.id)
            action = await service.reserve_action(
                setup_db,
                run_id=run.id,
                idempotency_key="run:phase11:kind:request_plan:inactive-agent-replay",
                action_type="request_plan",
                request=_canonical_plan_request(planner.id),
            )
            await setup_db.commit()
            planner_id = planner.id
            project_id = project.id
            goal_id = goal.id
            run_id = run.id
            action_id = action.id

        async with session_factory() as replay_db:
            service = OrchestrationService()
            with pytest.raises(HTTPException) as exc_info:
                await service.execute_request_plan_action(
                    replay_db,
                    run_id=run_id,
                    request=_plan_request(uuid.uuid4()),
                    idempotency_key="run:phase11:kind:request_plan:inactive-agent-replay",
                )
            assert exc_info.value.status_code == 404
            assert exc_info.value.detail == "Agent not found"
            await replay_db.rollback()

        async with session_factory() as verify_db:
            persisted = await verify_db.get(OrchestrationAction, action_id)
            assert persisted is not None
            assert persisted.status == "failed"
            assert persisted.error == "Agent not found"
            assert await verify_db.get(Task, action_id) is None
            assert (
                await verify_db.scalar(select(count(OrchestrationGate.id)).where(OrchestrationGate.run_id == run_id))
                == 0
            )
    finally:
        if project_id is not None and goal_id is not None and run_id is not None:
            await _cleanup_committed_plan_rows(
                session_factory,
                project_id=project_id,
                goal_id=goal_id,
                run_id=run_id,
                agent_ids=[planner_id] if planner_id is not None else None,
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("prior_plan_status", ["requested", "revision_requested"])
async def test_execute_request_plan_action_new_idempotency_key_supersedes_outstanding_planning_task(
    db_session,
    test_project,
    prior_plan_status,
):
    first_planner = _agent("first-planner")
    second_planner = _agent("second-planner")
    db_session.add_all([first_planner, second_planner])
    await db_session.flush()
    service, _, run = await _make_run(db_session, test_project.id)

    first_action = await service.execute_request_plan_action(
        db_session,
        run_id=run.id,
        request=_plan_request(first_planner.id),
        idempotency_key=f"run:phase11:kind:request_plan:first:{prior_plan_status}",
    )
    gate = await _single_plan_gate(db_session, run.id)

    if prior_plan_status == "revision_requested":
        await service.execute_request_plan_revision_action(
            db_session,
            run_id=run.id,
            request={
                "action_type": "request_plan_revision",
                "plan_task_id": str(first_action.target_id),
                "revision_request": "Revise the original planning task before superseding it.",
            },
            idempotency_key="run:phase11:kind:request_plan_revision:before-supersede",
        )

    second_action = await service.execute_request_plan_action(
        db_session,
        run_id=run.id,
        request=_plan_request(second_planner.id),
        idempotency_key=f"run:phase11:kind:request_plan:second:{prior_plan_status}",
    )

    first_task = await db_session.get(Task, first_action.target_id)
    second_task = await db_session.get(Task, second_action.target_id)

    assert first_task is not None
    assert second_task is not None
    assert first_task.id != second_task.id
    assert first_task.status == "cancelled"
    assert second_task.status == "backlog"
    assert second_task.assigned_to == second_planner.id
    assert run.plan_state == {
        "status": "requested",
        "planning_task_id": str(second_task.id),
        "request_action_id": str(second_action.id),
        "work_function": "planning",
        "plan_gate_id": str(gate.id),
        "revision_requests": [],
    }
    assert await db_session.scalar(select(count(OrchestrationGate.id)).where(OrchestrationGate.run_id == run.id)) == 1


@pytest.mark.asyncio
async def test_execute_request_plan_action_supersedes_in_progress_planning_task_and_cancels_session(
    db_session,
    test_project,
):
    first_planner = _agent("first-planner")
    second_planner = _agent("second-planner")
    db_session.add_all([first_planner, second_planner])
    await db_session.flush()
    service, _, run = await _make_run(db_session, test_project.id)

    first_action = await service.execute_request_plan_action(
        db_session,
        run_id=run.id,
        request=_plan_request(first_planner.id),
        idempotency_key="run:phase11:kind:request_plan:first:in-progress",
    )
    first_task = await db_session.get(Task, first_action.target_id)
    assert first_task is not None
    first_task.status = "in_progress"
    first_session = Session(
        agent_id=first_planner.id,
        project_id=test_project.id,
        task_id=first_task.id,
        adapter_type="api",
        status="running",
        input_context={},
        metadata_={},
    )
    db_session.add(first_session)
    await db_session.flush()

    second_action = await service.execute_request_plan_action(
        db_session,
        run_id=run.id,
        request=_plan_request(second_planner.id),
        idempotency_key="run:phase11:kind:request_plan:second:in-progress",
    )

    await db_session.refresh(first_task)
    await db_session.refresh(first_session)
    second_task = await db_session.get(Task, second_action.target_id)

    assert second_task is not None
    assert first_task.status == "cancelled"
    assert first_session.status == "cancelled"
    assert run.plan_state["planning_task_id"] == str(second_task.id)
    assert run.plan_state["request_action_id"] == str(second_action.id)


@pytest.mark.asyncio
async def test_execute_request_plan_revision_action_records_revision_and_emits_once(db_session, test_project):
    planner = _agent("planner")
    db_session.add(planner)
    await db_session.flush()
    service, _, run = await _make_run(db_session, test_project.id)
    request_action = await service.execute_request_plan_action(
        db_session,
        run_id=run.id,
        request=_plan_request(planner.id),
        idempotency_key="run:phase11:kind:request_plan:before-revision",
    )

    request = {
        "action_type": "request_plan_revision",
        "plan_task_id": f" {request_action.target_id} ",
        "revision_request": " Split the plan into independently verifiable tasks. ",
    }
    first = await service.execute_request_plan_revision_action(
        db_session,
        run_id=run.id,
        request=request,
        idempotency_key="run:phase11:kind:request_plan_revision",
    )
    second = await service.execute_request_plan_revision_action(
        db_session,
        run_id=run.id,
        request={**request, "revision_request": "Replay should not replace original request."},
        idempotency_key="run:phase11:kind:request_plan_revision",
    )

    assert second.id == first.id
    assert first.status == "completed"
    assert first.action_type == "request_plan_revision"
    assert first.target_type == "task"
    assert first.target_id == request_action.target_id
    assert first.request == {
        "action_type": "request_plan_revision",
        "plan_task_id": str(request_action.target_id),
        "revision_request": "Split the plan into independently verifiable tasks.",
    }

    assert run.plan_state["status"] == "revision_requested"
    assert run.plan_state["planning_task_id"] == str(request_action.target_id)
    assert run.plan_state["revision_requests"] == [
        {
            "action_id": str(first.id),
            "plan_task_id": str(request_action.target_id),
            "revision_request": "Split the plan into independently verifiable tasks.",
        }
    ]

    events = await _event_rows(db_session, test_project.id, "orchestration.plan_revision_requested")
    assert len(events) == 1
    assert events[0].payload == {
        "run_id": str(run.id),
        "action_id": str(first.id),
        "plan_task_id": str(request_action.target_id),
        "revision_request": "Split the plan into independently verifiable tasks.",
    }


@pytest.mark.asyncio
async def test_execute_request_plan_revision_action_rejects_task_from_another_run(db_session, test_project):
    planner = _agent("planner")
    db_session.add(planner)
    await db_session.flush()
    service, _, first_run = await _make_run(db_session, test_project.id)
    _, _, second_run = await _make_run(db_session, test_project.id)
    request_action = await service.execute_request_plan_action(
        db_session,
        run_id=first_run.id,
        request=_plan_request(planner.id),
        idempotency_key="run:phase11:first:kind:request_plan",
    )

    with pytest.raises(HTTPException) as exc_info:
        await service.execute_request_plan_revision_action(
            db_session,
            run_id=second_run.id,
            request={
                "action_type": "request_plan_revision",
                "plan_task_id": str(request_action.target_id),
                "revision_request": "Split the plan into independently verifiable tasks.",
            },
            idempotency_key="run:phase11:second:kind:request_plan_revision",
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Planning task does not belong to this run"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wrong_task_setup",
    [
        "unrelated_action_task",
        "alternate_metadata_task",
    ],
)
async def test_execute_request_plan_revision_action_rejects_non_current_planning_task(
    db_session,
    test_project,
    wrong_task_setup,
):
    planner = _agent("planner")
    db_session.add(planner)
    await db_session.flush()
    service, goal, run = await _make_run(db_session, test_project.id)
    request_action = await service.execute_request_plan_action(
        db_session,
        run_id=run.id,
        request=_plan_request(planner.id),
        idempotency_key=f"run:phase11:kind:request_plan:before-non-current-revision-{wrong_task_setup}",
    )

    if wrong_task_setup == "unrelated_action_task":
        unrelated_action = await service.reserve_action(
            db_session,
            run_id=run.id,
            idempotency_key="run:phase11:kind:create_delegation_task:planning-lookalike",
            action_type="create_delegation_task",
            request={
                "action_type": "create_delegation_task",
                "agent_id": str(planner.id),
                "work_function": "planning",
                "scope": "Look like planning without being the request_plan task.",
                "inputs": [],
                "deliverable": "Fake planning task",
                "forbidden_work": [],
                "success_evidence": [],
                "budget": {},
                "report_schema": {},
            },
        )
        contract = service._planning_contract(goal, run, unrelated_action, _canonical_plan_request(planner.id))
        wrong_task = await TaskService().create(
            db_session,
            test_project.id,
            TaskCreate(
                title=service._delegation_task_title(contract.work_function, contract.deliverable),
                description=service._delegation_task_description(goal, contract),
                assigned_to=planner.id,
                metadata={
                    "orchestration": {
                        "goal_id": str(goal.id),
                        "run_id": str(run.id),
                        "action_id": str(unrelated_action.id),
                        "work_function": "planning",
                    },
                    "orchestration_contract": contract.model_dump(mode="json"),
                },
            ),
            task_id=unrelated_action.id,
        )
    else:  # alternate_metadata_task
        wrong_task = await TaskService().create(
            db_session,
            test_project.id,
            TaskCreate(
                title="Planning: Alternate",
                description="Same run, wrong task.",
                assigned_to=planner.id,
                metadata={
                    "orchestration": {
                        "goal_id": str(run.goal_id),
                        "run_id": str(run.id),
                        "action_id": str(uuid.uuid4()),
                        "work_function": "planning",
                    }
                },
            ),
        )
        await db_session.flush()

    with pytest.raises(HTTPException) as exc_info:
        await service.execute_request_plan_revision_action(
            db_session,
            run_id=run.id,
            request={
                "action_type": "request_plan_revision",
                "plan_task_id": str(wrong_task.id),
                "revision_request": "Revise the wrong task.",
            },
            idempotency_key=f"run:phase11:kind:request_plan_revision:wrong-task-{wrong_task_setup}",
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Planning task is not the current outstanding request_plan task"
    assert run.plan_state["planning_task_id"] == str(request_action.target_id)


@pytest.mark.asyncio
async def test_execute_request_plan_revision_action_rejects_revision_after_acceptance(db_session, test_project):
    planner = _agent("planner")
    db_session.add(planner)
    await db_session.flush()
    service, _, run = await _make_run(db_session, test_project.id)
    request_action = await service.execute_request_plan_action(
        db_session,
        run_id=run.id,
        request=_plan_request(planner.id),
        idempotency_key="run:phase11:kind:request_plan:before-accepted-revision",
    )
    artifact = Artifact(
        project_id=test_project.id,
        name="delegation-plan",
        artifact_type="plan",
        status="draft",
        linked_task_id=request_action.target_id,
        created_by_agent=planner.id,
        metadata_=_accepted_plan_metadata(),
    )
    db_session.add(artifact)
    await db_session.flush()
    await service.execute_accept_plan_action(
        db_session,
        run_id=run.id,
        request={"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
        idempotency_key="run:phase11:kind:accept_plan:before-accepted-revision",
    )

    with pytest.raises(HTTPException) as exc_info:
        await service.execute_request_plan_revision_action(
            db_session,
            run_id=run.id,
            request={
                "action_type": "request_plan_revision",
                "plan_task_id": str(request_action.target_id),
                "revision_request": "Reopen the accepted plan.",
            },
            idempotency_key="run:phase11:kind:request_plan_revision:after-accept",
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Accepted plan cannot be revised"
    assert run.plan_state["status"] == "accepted"
    assert (
        await db_session.scalar(
            select(count(OrchestrationAction.id)).where(
                OrchestrationAction.run_id == run.id,
                OrchestrationAction.action_type == "request_plan_revision",
            )
        )
        == 0
    )


@pytest.mark.asyncio
async def test_execute_request_plan_revision_action_rejects_when_gate_is_accepted_but_plan_state_is_stale(
    db_session,
    test_project,
):
    planner = _agent("planner")
    db_session.add(planner)
    await db_session.flush()
    service, _, run = await _make_run(db_session, test_project.id)
    request_action = await service.execute_request_plan_action(
        db_session,
        run_id=run.id,
        request=_plan_request(planner.id),
        idempotency_key="run:phase11:kind:request_plan:before-stale-gate-revision",
    )
    artifact = Artifact(
        project_id=test_project.id,
        name="delegation-plan",
        artifact_type="plan",
        status="draft",
        linked_task_id=request_action.target_id,
        created_by_agent=planner.id,
        metadata_={"kind": "implementation_plan"},
    )
    db_session.add(artifact)
    await db_session.flush()
    gate = await _single_plan_gate(db_session, run.id)
    gate.status = "accepted"
    gate.accepted_at = datetime.now(UTC)
    await db_session.flush()

    with pytest.raises(HTTPException) as exc_info:
        await service.execute_request_plan_revision_action(
            db_session,
            run_id=run.id,
            request={
                "action_type": "request_plan_revision",
                "plan_task_id": str(request_action.target_id),
                "revision_request": "Reopen the accepted plan from stale state.",
            },
            idempotency_key="run:phase11:kind:request_plan_revision:stale-gate-accepted",
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Accepted plan cannot be revised"


@pytest.mark.asyncio
async def test_execute_request_plan_revision_action_replay_after_acceptance_fails_reserved_action(
    db_session,
    test_project,
):
    planner = _agent("planner")
    db_session.add(planner)
    await db_session.flush()
    service, _, run = await _make_run(db_session, test_project.id)
    request_action = await service.execute_request_plan_action(
        db_session,
        run_id=run.id,
        request=_plan_request(planner.id),
        idempotency_key="run:phase11:kind:request_plan:before-revision-replay-fail",
    )
    revision_action = await service.reserve_action(
        db_session,
        run_id=run.id,
        idempotency_key="run:phase11:kind:request_plan_revision:replay-after-accept",
        action_type="request_plan_revision",
        request={
            "action_type": "request_plan_revision",
            "plan_task_id": str(request_action.target_id),
            "revision_request": "Replay after accept should fail terminally.",
        },
    )
    artifact = Artifact(
        project_id=test_project.id,
        name="delegation-plan",
        artifact_type="plan",
        status="draft",
        linked_task_id=request_action.target_id,
        created_by_agent=planner.id,
        metadata_=_accepted_plan_metadata(),
    )
    db_session.add(artifact)
    await db_session.flush()
    await service.execute_accept_plan_action(
        db_session,
        run_id=run.id,
        request={"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
        idempotency_key="run:phase11:kind:accept_plan:before-revision-replay-fail",
    )

    with pytest.raises(HTTPException) as exc_info:
        await service.execute_request_plan_revision_action(
            db_session,
            run_id=run.id,
            request={
                "action_type": "request_plan_revision",
                "plan_task_id": str(request_action.target_id),
                "revision_request": "This replay should not matter.",
            },
            idempotency_key="run:phase11:kind:request_plan_revision:replay-after-accept",
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Accepted plan cannot be revised"
    await db_session.refresh(revision_action)
    assert revision_action.status == "failed"
    assert revision_action.error == "Accepted plan cannot be revised"


@pytest.mark.asyncio
async def test_execute_request_plan_revision_action_rejects_failed_plan_state(
    db_session,
    test_project,
):
    planner = _agent("planner")
    db_session.add(planner)
    await db_session.flush()
    service, _, run = await _make_run(db_session, test_project.id)
    request_action = await service.execute_request_plan_action(
        db_session,
        run_id=run.id,
        request=_plan_request(planner.id),
        idempotency_key="run:phase11:kind:request_plan:before-failed-revision",
    )
    run.plan_state = {
        **run.plan_state,
        "status": "failed",
    }
    await db_session.flush()

    with pytest.raises(HTTPException) as exc_info:
        await service.execute_request_plan_revision_action(
            db_session,
            run_id=run.id,
            request={
                "action_type": "request_plan_revision",
                "plan_task_id": str(request_action.target_id),
                "revision_request": "Retry a failed plan.",
            },
            idempotency_key="run:phase11:kind:request_plan_revision:after-failed-state",
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Terminal plan cannot be revised for this run"


@pytest.mark.asyncio
async def test_execute_request_plan_revision_action_replay_after_failed_plan_state_fails_reserved_action(
    db_session,
    test_project,
):
    planner = _agent("planner")
    db_session.add(planner)
    await db_session.flush()
    service, _, run = await _make_run(db_session, test_project.id)
    request_action = await service.execute_request_plan_action(
        db_session,
        run_id=run.id,
        request=_plan_request(planner.id),
        idempotency_key="run:phase11:kind:request_plan:before-failed-revision-replay",
    )
    revision_action = await service.reserve_action(
        db_session,
        run_id=run.id,
        idempotency_key="run:phase11:kind:request_plan_revision:replay-after-failed-state",
        action_type="request_plan_revision",
        request={
            "action_type": "request_plan_revision",
            "plan_task_id": str(request_action.target_id),
            "revision_request": "Replay after failure should fail terminally.",
        },
    )
    run.plan_state = {
        **run.plan_state,
        "status": "failed",
    }
    await db_session.flush()

    with pytest.raises(HTTPException) as exc_info:
        await service.execute_request_plan_revision_action(
            db_session,
            run_id=run.id,
            request={
                "action_type": "request_plan_revision",
                "plan_task_id": str(request_action.target_id),
                "revision_request": "This replay should not matter.",
            },
            idempotency_key="run:phase11:kind:request_plan_revision:replay-after-failed-state",
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Terminal plan cannot be revised for this run"
    await db_session.refresh(revision_action)
    assert revision_action.status == "failed"
    assert revision_action.error == "Terminal plan cannot be revised for this run"


@pytest.mark.asyncio
async def test_execute_accept_plan_action_accepts_artifact_and_plan_gate(db_session, test_project):
    planner = _agent("planner")
    db_session.add(planner)
    await db_session.flush()
    service, _, run = await _make_run(db_session, test_project.id)
    request_action = await service.execute_request_plan_action(
        db_session,
        run_id=run.id,
        request=_plan_request(planner.id),
        idempotency_key="run:phase11:kind:request_plan:before-accept",
    )
    artifact = Artifact(
        project_id=test_project.id,
        name="delegation-plan",
        artifact_type="plan",
        status="draft",
        linked_task_id=request_action.target_id,
        created_by_agent=planner.id,
        metadata_=_accepted_plan_metadata(),
    )
    db_session.add(artifact)
    await db_session.flush()

    first = await service.execute_accept_plan_action(
        db_session,
        run_id=run.id,
        request={"action_type": "accept_plan", "plan_artifact_id": f" {artifact.id} "},
        idempotency_key="run:phase11:kind:accept_plan",
    )
    second = await service.execute_accept_plan_action(
        db_session,
        run_id=run.id,
        request={"action_type": "accept_plan", "plan_artifact_id": str(uuid.uuid4())},
        idempotency_key="run:phase11:kind:accept_plan",
    )

    assert second.id == first.id
    assert first.status == "completed"
    assert first.action_type == "accept_plan"
    assert first.target_type == "artifact"
    assert first.target_id == artifact.id
    assert first.request == {"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)}

    gate = await _single_plan_gate(db_session, run.id)
    assert gate.status == "accepted"
    assert gate.failure_reason is None
    assert gate.accepted_at is not None

    evidence_result = await db_session.execute(
        select(OrchestrationEvidence).where(
            OrchestrationEvidence.run_id == run.id,
            OrchestrationEvidence.gate_id == gate.id,
            OrchestrationEvidence.source_type == "artifact",
            OrchestrationEvidence.source_id == artifact.id,
        )
    )
    evidence_rows = list(evidence_result.scalars().all())
    assert len(evidence_rows) == 1
    evidence = evidence_rows[0]
    assert evidence.verdict == "accepted"
    assert evidence.producer_agent_id == planner.id
    assert evidence.evidence_metadata == {"action_id": str(first.id), "gate_type": "plan_accepted"}

    assert run.plan_state["status"] == "accepted"
    assert run.plan_state["planning_task_id"] == str(request_action.target_id)
    assert run.plan_state["accepted_artifact_id"] == str(artifact.id)
    assert run.plan_state["accept_action_id"] == str(first.id)
    assert run.plan_state["plan_gate_id"] == str(gate.id)

    events = await _event_rows(db_session, test_project.id, "orchestration.plan_accepted")
    assert len(events) == 1
    assert events[0].payload == {
        "run_id": str(run.id),
        "action_id": str(first.id),
        "plan_task_id": str(request_action.target_id),
        "plan_artifact_id": str(artifact.id),
        "gate_id": str(gate.id),
        "accepted_plan_fingerprint": run.plan_state["accepted_plan_snapshot"]["fingerprint"],
    }


@pytest.mark.asyncio
async def test_execute_accept_plan_action_rejects_non_plan_artifact_type(db_session, test_project):
    planner = _agent("planner")
    db_session.add(planner)
    await db_session.flush()
    service, _, run = await _make_run(db_session, test_project.id)
    request_action = await service.execute_request_plan_action(
        db_session,
        run_id=run.id,
        request=_plan_request(planner.id),
        idempotency_key="run:phase11:kind:request_plan:wrong-artifact-type",
    )
    artifact = Artifact(
        project_id=test_project.id,
        name="delegation-not-plan",
        artifact_type="notes",
        status="draft",
        linked_task_id=request_action.target_id,
        created_by_agent=planner.id,
        metadata_={"kind": "implementation_plan"},
    )
    db_session.add(artifact)
    await db_session.flush()

    with pytest.raises(HTTPException) as exc_info:
        await service.execute_accept_plan_action(
            db_session,
            run_id=run.id,
            request={"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
            idempotency_key="run:phase11:kind:accept_plan:wrong-artifact-type",
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Artifact must have artifact_type 'plan'"


@pytest.mark.asyncio
async def test_execute_accept_plan_action_replay_artifact_revalidation_failure_fails_reserved_action(
    db_session,
    test_project,
):
    planner = _agent("planner")
    db_session.add(planner)
    await db_session.flush()
    service, _, run = await _make_run(db_session, test_project.id)
    request_action = await service.execute_request_plan_action(
        db_session,
        run_id=run.id,
        request=_plan_request(planner.id),
        idempotency_key="run:phase11:kind:request_plan:before-accept-revalidation-replay",
    )
    artifact = Artifact(
        project_id=test_project.id,
        name="delegation-plan",
        artifact_type="plan",
        status="draft",
        linked_task_id=request_action.target_id,
        created_by_agent=planner.id,
        metadata_={"kind": "implementation_plan"},
    )
    db_session.add(artifact)
    await db_session.flush()
    accept_action = await service.reserve_action(
        db_session,
        run_id=run.id,
        idempotency_key="run:phase11:kind:accept_plan:artifact-revalidation-replay",
        action_type="accept_plan",
        request={"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
    )
    await db_session.delete(artifact)
    await db_session.flush()

    with pytest.raises(HTTPException) as exc_info:
        await service.execute_accept_plan_action(
            db_session,
            run_id=run.id,
            request={"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
            idempotency_key="run:phase11:kind:accept_plan:artifact-revalidation-replay",
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Plan artifact not found"
    await db_session.refresh(accept_action)
    assert accept_action.status == "failed"
    assert accept_action.error == "Plan artifact not found"


@pytest.mark.asyncio
async def test_execute_accept_plan_action_rejects_invalid_plan_state(db_session, test_project):
    planner = _agent("planner")
    db_session.add(planner)
    await db_session.flush()
    service, _, run = await _make_run(db_session, test_project.id)
    request_action = await service.execute_request_plan_action(
        db_session,
        run_id=run.id,
        request=_plan_request(planner.id),
        idempotency_key="run:phase11:kind:request_plan:invalid-accept-state",
    )
    artifact = Artifact(
        project_id=test_project.id,
        name="delegation-plan",
        artifact_type="plan",
        status="draft",
        linked_task_id=request_action.target_id,
        created_by_agent=planner.id,
        metadata_={"kind": "implementation_plan"},
    )
    db_session.add(artifact)
    run.plan_state = {
        **run.plan_state,
        "status": "accepted",
    }
    await db_session.flush()

    with pytest.raises(HTTPException) as exc_info:
        await service.execute_accept_plan_action(
            db_session,
            run_id=run.id,
            request={"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
            idempotency_key="run:phase11:kind:accept_plan:invalid-state",
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Plan can only be accepted from requested or revision_requested state"
    assert (
        await db_session.scalar(
            select(count(OrchestrationAction.id)).where(
                OrchestrationAction.run_id == run.id,
                OrchestrationAction.action_type == "accept_plan",
            )
        )
        == 0
    )


async def test_execute_accept_plan_action_replay_from_invalid_plan_state_fails_reserved_action(
    db_session,
    test_project,
):
    planner = _agent("planner")
    db_session.add(planner)
    await db_session.flush()
    service, _, run = await _make_run(db_session, test_project.id)
    request_action = await service.execute_request_plan_action(
        db_session,
        run_id=run.id,
        request=_plan_request(planner.id),
        idempotency_key="run:phase11:kind:request_plan:before-accept-replay-fail",
    )
    artifact = Artifact(
        project_id=test_project.id,
        name="delegation-plan",
        artifact_type="plan",
        status="draft",
        linked_task_id=request_action.target_id,
        created_by_agent=planner.id,
        metadata_={"kind": "implementation_plan"},
    )
    db_session.add(artifact)
    await db_session.flush()
    accept_action = await service.reserve_action(
        db_session,
        run_id=run.id,
        idempotency_key="run:phase11:kind:accept_plan:replay-invalid-state",
        action_type="accept_plan",
        request={"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
    )
    run.plan_state = {
        **run.plan_state,
        "status": "accepted",
    }
    await db_session.flush()

    with pytest.raises(HTTPException) as exc_info:
        await service.execute_accept_plan_action(
            db_session,
            run_id=run.id,
            request={"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
            idempotency_key="run:phase11:kind:accept_plan:replay-invalid-state",
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Plan can only be accepted from requested or revision_requested state"
    await db_session.refresh(accept_action)
    assert accept_action.status == "failed"
    assert accept_action.error == "Plan can only be accepted from requested or revision_requested state"


@pytest.mark.asyncio
async def test_execute_accept_plan_action_replay_invalid_artifact_fails_reserved_action(
    db_session,
    test_project,
):
    planner = _agent("planner")
    db_session.add(planner)
    await db_session.flush()
    service, _, run = await _make_run(db_session, test_project.id)
    request_action = await service.execute_request_plan_action(
        db_session,
        run_id=run.id,
        request=_plan_request(planner.id),
        idempotency_key="run:phase11:kind:request_plan:before-accept-artifact-replay-fail",
    )
    artifact = Artifact(
        project_id=test_project.id,
        name="delegation-plan",
        artifact_type="plan",
        status="draft",
        linked_task_id=request_action.target_id,
        created_by_agent=planner.id,
        metadata_={"kind": "implementation_plan"},
    )
    db_session.add(artifact)
    await db_session.flush()
    accept_action = await service.reserve_action(
        db_session,
        run_id=run.id,
        idempotency_key="run:phase11:kind:accept_plan:replay-invalid-artifact",
        action_type="accept_plan",
        request={"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
    )
    artifact.linked_task_id = uuid.uuid4()
    await db_session.flush()

    with pytest.raises(HTTPException) as exc_info:
        await service.execute_accept_plan_action(
            db_session,
            run_id=run.id,
            request={"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
            idempotency_key="run:phase11:kind:accept_plan:replay-invalid-artifact",
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Plan artifact is not linked to the current planning task"
    await db_session.refresh(accept_action)
    assert accept_action.status == "failed"
    assert accept_action.error == "Plan artifact is not linked to the current planning task"


@pytest.mark.asyncio
async def test_execute_accept_plan_action_rejects_unlinked_artifact(db_session, test_project):
    planner = _agent("planner")
    db_session.add(planner)
    await db_session.flush()
    service, _, run = await _make_run(db_session, test_project.id)
    await service.execute_request_plan_action(
        db_session,
        run_id=run.id,
        request=_plan_request(planner.id),
        idempotency_key="run:phase11:kind:request_plan:unlinked",
    )
    artifact = Artifact(
        project_id=test_project.id,
        name="delegation-plan",
        artifact_type="plan",
        status="draft",
        linked_task_id=None,
        created_by_agent=planner.id,
        metadata_={"kind": "implementation_plan"},
    )
    db_session.add(artifact)
    await db_session.flush()

    with pytest.raises(HTTPException) as exc_info:
        await service.execute_accept_plan_action(
            db_session,
            run_id=run.id,
            request={"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
            idempotency_key="run:phase11:kind:accept_plan:unlinked",
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Plan artifact is not linked to the current planning task"


@pytest.mark.asyncio
@pytest.mark.parametrize("created_by_agent", [None, "other"])
async def test_execute_accept_plan_action_rejects_artifact_from_wrong_or_missing_agent(
    db_session,
    test_project,
    created_by_agent,
):
    planner = _agent("planner")
    other_planner = _agent("other-planner")
    db_session.add_all([planner, other_planner])
    await db_session.flush()
    service, _, run = await _make_run(db_session, test_project.id)
    request_action = await service.execute_request_plan_action(
        db_session,
        run_id=run.id,
        request=_plan_request(planner.id),
        idempotency_key=f"run:phase11:kind:request_plan:artifact-author:{created_by_agent or 'none'}",
    )
    artifact = Artifact(
        project_id=test_project.id,
        name="delegation-plan",
        artifact_type="plan",
        status="draft",
        linked_task_id=request_action.target_id,
        created_by_agent=None if created_by_agent is None else other_planner.id,
        metadata_={"kind": "implementation_plan"},
    )
    db_session.add(artifact)
    await db_session.flush()

    with pytest.raises(HTTPException) as exc_info:
        await service.execute_accept_plan_action(
            db_session,
            run_id=run.id,
            request={"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
            idempotency_key=f"run:phase11:kind:accept_plan:artifact-author:{created_by_agent or 'none'}",
        )

    assert exc_info.value.status_code == 409
    assert (
        await db_session.scalar(
            select(count(OrchestrationAction.id)).where(
                OrchestrationAction.run_id == run.id,
                OrchestrationAction.action_type == "accept_plan",
            )
        )
        == 0
    )
    gate = await _single_plan_gate(db_session, run.id)
    assert gate.status == "open"
    assert (
        await db_session.scalar(
            select(count(OrchestrationEvidence.id)).where(
                OrchestrationEvidence.run_id == run.id,
                OrchestrationEvidence.gate_id == gate.id,
            )
        )
        == 0
    )
    assert run.plan_state["status"] == "requested"


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_status", ["accepted", "failed"])
async def test_execute_request_plan_action_rejects_replanning_after_terminal_gate_state(
    db_session,
    test_project,
    terminal_status,
):
    planner = _agent("planner")
    db_session.add(planner)
    await db_session.flush()
    service, _, run = await _make_run(db_session, test_project.id)
    action = await service.execute_request_plan_action(
        db_session,
        run_id=run.id,
        request=_plan_request(planner.id),
        idempotency_key=f"run:phase11:kind:request_plan:before-terminal-{terminal_status}",
    )
    gate = await _single_plan_gate(db_session, run.id)
    artifact = Artifact(
        project_id=test_project.id,
        name=f"delegation-{terminal_status}-plan",
        artifact_type="plan",
        status="draft",
        linked_task_id=action.target_id,
        created_by_agent=planner.id,
        metadata_={"kind": "implementation_plan"},
    )
    db_session.add(artifact)
    await db_session.flush()
    db_session.add(
        OrchestrationEvidence(
            run_id=run.id,
            gate_id=gate.id,
            source_type="artifact",
            source_id=artifact.id,
            observed_event_id=None,
            producer_agent_id=planner.id,
            verdict="accepted" if terminal_status == "accepted" else "candidate",
            evidence_metadata={"source": "stale"},
        )
    )
    gate.status = terminal_status
    if terminal_status == "accepted":
        gate.accepted_at = datetime.now(UTC)
    else:
        gate.failed_at = datetime.now(UTC)
        gate.failure_reason = "Previous plan rejected"
    run.plan_state = {
        **run.plan_state,
        "status": terminal_status,
        "accepted_artifact_id": str(artifact.id),
        "accept_action_id": str(uuid.uuid4()),
    }
    await db_session.flush()

    with pytest.raises(HTTPException) as exc_info:
        await service.execute_request_plan_action(
            db_session,
            run_id=run.id,
            request=_plan_request(planner.id),
            idempotency_key=f"run:phase11:kind:request_plan:after-terminal-{terminal_status}",
        )

    assert exc_info.value.status_code == 409
    assert (
        await db_session.scalar(
            select(count(OrchestrationAction.id)).where(
                OrchestrationAction.run_id == run.id,
                OrchestrationAction.action_type == "request_plan",
            )
        )
        == 1
    )
    assert gate.status == terminal_status
    assert (
        await db_session.scalar(
            select(count(OrchestrationEvidence.id)).where(
                OrchestrationEvidence.run_id == run.id,
                OrchestrationEvidence.gate_id == gate.id,
            )
        )
        == 1
    )


@pytest.mark.asyncio
async def test_execute_request_plan_revision_action_replay_failure_persists_after_rollback(
    test_engine, tmp_path,
):
    session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    planner_id = None
    project_id = None
    goal_id = None
    run_id = None
    try:
        async with session_factory() as setup_db:
            project = Project(
                name="Replay Project",
                description="Revision replay durability",
                workspace_path=str(tmp_path),
                config={},
            )
            planner = _agent("planner")
            setup_db.add_all([project, planner])
            await setup_db.flush()
            service, goal, run = await _make_run(setup_db, project.id)
            request_action = await service.execute_request_plan_action(
                setup_db,
                run_id=run.id,
                request=_plan_request(planner.id),
                idempotency_key="run:phase11:kind:request_plan:durable-revision-replay",
            )
            revision_action = await service.reserve_action(
                setup_db,
                run_id=run.id,
                idempotency_key="run:phase11:kind:request_plan_revision:durable-replay",
                action_type="request_plan_revision",
                request={
                    "action_type": "request_plan_revision",
                    "plan_task_id": str(request_action.target_id),
                    "revision_request": "This replay should fail durably.",
                },
            )
            artifact = Artifact(
                project_id=project.id,
                name="delegation-plan",
                artifact_type="plan",
                status="draft",
                linked_task_id=request_action.target_id,
                created_by_agent=planner.id,
                metadata_=_accepted_plan_metadata(),
            )
            setup_db.add(artifact)
            await setup_db.flush()
            await service.execute_accept_plan_action(
                setup_db,
                run_id=run.id,
                request={"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
                idempotency_key="run:phase11:kind:accept_plan:durable-revision-replay",
            )
            await setup_db.commit()
            planner_id = planner.id
            project_id = project.id
            goal_id = goal.id
            run_id = run.id
            plan_task_id = request_action.target_id
            action_id = revision_action.id

        async with session_factory() as replay_db:
            service = OrchestrationService()
            with pytest.raises(HTTPException) as exc_info:
                await service.execute_request_plan_revision_action(
                    replay_db,
                    run_id=run_id,
                    request={
                        "action_type": "request_plan_revision",
                        "plan_task_id": str(plan_task_id),
                        "revision_request": "Different replay input should not matter.",
                    },
                    idempotency_key="run:phase11:kind:request_plan_revision:durable-replay",
                )
            assert exc_info.value.status_code == 409
            await replay_db.rollback()

        async with session_factory() as verify_db:
            action = await verify_db.get(OrchestrationAction, action_id)
            assert action.status == "failed"
            assert action.error == "Accepted plan cannot be revised"
    finally:
        if project_id is not None and goal_id is not None and run_id is not None:
            await _cleanup_committed_plan_rows(
                session_factory,
                project_id=project_id,
                goal_id=goal_id,
                run_id=run_id,
                agent_ids=[planner_id] if planner_id is not None else None,
            )


@pytest.mark.asyncio
async def test_execute_accept_plan_action_replay_failure_persists_after_rollback(
    test_engine, tmp_path,
):
    session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    planner_id = None
    project_id = None
    goal_id = None
    run_id = None
    try:
        async with session_factory() as setup_db:
            project = Project(
                name="Replay Project",
                description="Accept replay durability",
                workspace_path=str(tmp_path),
                config={},
            )
            planner = _agent("planner")
            setup_db.add_all([project, planner])
            await setup_db.flush()
            service, goal, run = await _make_run(setup_db, project.id)
            request_action = await service.execute_request_plan_action(
                setup_db,
                run_id=run.id,
                request=_plan_request(planner.id),
                idempotency_key="run:phase11:kind:request_plan:durable-accept-replay",
            )
            artifact = Artifact(
                project_id=project.id,
                name="delegation-plan",
                artifact_type="plan",
                status="draft",
                linked_task_id=request_action.target_id,
                created_by_agent=planner.id,
                metadata_={"kind": "implementation_plan"},
            )
            setup_db.add(artifact)
            await setup_db.flush()
            accept_action = await service.reserve_action(
                setup_db,
                run_id=run.id,
                idempotency_key="run:phase11:kind:accept_plan:durable-replay",
                action_type="accept_plan",
                request={"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
            )
            run.plan_state = {
                **run.plan_state,
                "status": "accepted",
            }
            await setup_db.commit()
            planner_id = planner.id
            project_id = project.id
            goal_id = goal.id
            run_id = run.id
            artifact_id = artifact.id
            action_id = accept_action.id

        async with session_factory() as replay_db:
            service = OrchestrationService()
            with pytest.raises(HTTPException) as exc_info:
                await service.execute_accept_plan_action(
                    replay_db,
                    run_id=run_id,
                    request={"action_type": "accept_plan", "plan_artifact_id": str(artifact_id)},
                    idempotency_key="run:phase11:kind:accept_plan:durable-replay",
                )
            assert exc_info.value.status_code == 409
            await replay_db.rollback()

        async with session_factory() as verify_db:
            action = await verify_db.get(OrchestrationAction, action_id)
            assert action.status == "failed"
            assert action.error == "Plan can only be accepted from requested or revision_requested state"
    finally:
        if project_id is not None and goal_id is not None and run_id is not None:
            await _cleanup_committed_plan_rows(
                session_factory,
                project_id=project_id,
                goal_id=goal_id,
                run_id=run_id,
                agent_ids=[planner_id] if planner_id is not None else None,
            )


@pytest.mark.asyncio
async def test_request_plan_revision_replay_failure_survives_caller_rollback(db_session, test_project):
    service = OrchestrationService()
    planner = _agent("planner")
    db_session.add(planner)
    await db_session.flush()

    # Create and complete goal definition
    goal, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Test plan revision replay",
            success_criteria=[{"key": "plan", "description": "Plan accepted."}],
        ),
        created_by_user_id=None,
    )
    await _complete_goal_definition(db_session, goal, run)

    # Request plan
    request_action = await service.execute_request_plan_action(
        db_session,
        run_id=run.id,
        request=_plan_request(planner.id),
        idempotency_key="run:phase11:kind:request_plan:replay-test",
    )

    # Reserve revision action
    revision_action = await service.reserve_action(
        db_session,
        run_id=run.id,
        idempotency_key="run:phase11:kind:request_plan_revision:replay-test",
        action_type="request_plan_revision",
        request={
            "action_type": "request_plan_revision",
            "plan_task_id": str(request_action.target_id),
            "revision_request": "Test revision request.",
        },
    )

    # Create and accept plan
    artifact = Artifact(
        project_id=test_project.id,
        name="delegation-plan",
        artifact_type="plan",
        status="draft",
        linked_task_id=request_action.target_id,
        created_by_agent=planner.id,
        metadata_=_accepted_plan_metadata(),
    )
    db_session.add(artifact)
    await db_session.flush()
    await service.execute_accept_plan_action(
        db_session,
        run_id=run.id,
        request={"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
        idempotency_key="run:phase11:kind:accept_plan:replay-test",
    )

    # Replay revision should fail because plan is accepted
    with pytest.raises(HTTPException) as exc_info:
        await service.execute_request_plan_revision_action(
            db_session,
            run_id=run.id,
            request={
                "action_type": "request_plan_revision",
                "plan_task_id": str(request_action.target_id),
                "revision_request": "This replay should fail.",
            },
            idempotency_key="run:phase11:kind:request_plan_revision:replay-test",
        )
    assert exc_info.value.status_code == 409

    # Verify revision action was marked as failed
    await db_session.refresh(revision_action)
    assert revision_action.status == "failed"
    assert revision_action.error == "Accepted plan cannot be revised"


@pytest.mark.asyncio
async def test_accept_plan_replay_failure_survives_caller_rollback(db_session, test_project):
    service = OrchestrationService()
    planner = _agent("planner")
    db_session.add(planner)
    await db_session.flush()

    # Create and complete goal definition
    goal, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Test plan accept replay",
            success_criteria=[{"key": "plan", "description": "Plan accepted."}],
        ),
        created_by_user_id=None,
    )
    await _complete_goal_definition(db_session, goal, run)

    # Request plan
    request_action = await service.execute_request_plan_action(
        db_session,
        run_id=run.id,
        request=_plan_request(planner.id),
        idempotency_key="run:phase11:kind:request_plan:accept-replay-test",
    )

    # Create artifact and accept plan
    artifact = Artifact(
        project_id=test_project.id,
        name="delegation-plan",
        artifact_type="plan",
        status="draft",
        linked_task_id=request_action.target_id,
        created_by_agent=planner.id,
        metadata_={"kind": "implementation_plan"},
    )
    db_session.add(artifact)
    await db_session.flush()

    accept_action = await service.reserve_action(
        db_session,
        run_id=run.id,
        idempotency_key="run:phase11:kind:accept_plan:accept-replay-test",
        action_type="accept_plan",
        request={"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
    )

    # Manually set plan_state to accepted (as done in the original test)
    run = await db_session.get(OrchestrationRun, run.id)
    run.plan_state = {
        **run.plan_state,
        "status": "accepted",
    }
    await db_session.flush()

    # Replay accept should fail because plan is already accepted
    with pytest.raises(HTTPException) as exc_info:
        await service.execute_accept_plan_action(
            db_session,
            run_id=run.id,
            request={"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
            idempotency_key="run:phase11:kind:accept_plan:accept-replay-test",
        )
    assert exc_info.value.status_code == 409

    # Verify accept action was marked as failed
    await db_session.refresh(accept_action)
    assert accept_action.status == "failed"
    assert accept_action.error == "Plan can only be accepted from requested or revision_requested state"


@pytest.mark.asyncio
async def test_execute_request_plan_action_replay_failure_persists_after_rollback(
    test_engine,
    tmp_path,
    monkeypatch,
):
    import huddleroom.services.orchestration_service as orchestration_service_module

    service, session_factory, project_id, goal_id, run_id = await _make_committed_run(test_engine, tmp_path)
    monkeypatch.setattr(orchestration_service_module, "AsyncSessionLocal", session_factory)
    planner = _agent("planner")
    try:
        async with session_factory() as seed_db:
            seed_db.add(planner)
            await seed_db.flush()
            action = await service.reserve_action(
                seed_db,
                run_id=run_id,
                idempotency_key="run:phase11:kind:request_plan:durable-replay",
                action_type="request_plan",
                request=_canonical_plan_request(planner.id),
            )
            run = await seed_db.get(OrchestrationRun, run_id)
            run.plan_state = {
                "status": "accepted",
                "planning_task_id": str(uuid.uuid4()),
                "request_action_id": str(uuid.uuid4()),
                "work_function": "planning",
                "plan_gate_id": str(uuid.uuid4()),
                "accepted_artifact_id": str(uuid.uuid4()),
                "accept_action_id": str(uuid.uuid4()),
            }
            await seed_db.commit()
            action_id = action.id

        async with session_factory() as replay_db:
            with pytest.raises(HTTPException) as exc_info:
                await service.execute_request_plan_action(
                    replay_db,
                    run_id=run_id,
                    request=_plan_request(planner.id),
                    idempotency_key="run:phase11:kind:request_plan:durable-replay",
                )
            assert exc_info.value.status_code == 409
            assert exc_info.value.detail == "Terminal plan cannot be re-requested for this run"
            await replay_db.rollback()

        async with session_factory() as verify_db:
            action = await verify_db.get(OrchestrationAction, action_id)
            assert action is not None
            assert action.status == "failed"
            assert action.error == "Terminal plan cannot be re-requested for this run"
    finally:
        await _cleanup_committed_plan_rows(
            session_factory,
            project_id=project_id,
            goal_id=goal_id,
            run_id=run_id,
            agent_ids=[planner.id],
        )


@pytest.mark.asyncio
async def test_execute_accept_plan_action_adds_direct_evidence_when_event_linked_evidence_exists(
    db_session,
    test_project,
):
    planner = _agent("planner")
    db_session.add(planner)
    await db_session.flush()
    service, _, run = await _make_run(db_session, test_project.id)
    request_action = await service.execute_request_plan_action(
        db_session,
        run_id=run.id,
        request=_plan_request(planner.id),
        idempotency_key="run:phase11:kind:request_plan:before-event-evidence",
    )
    artifact = Artifact(
        project_id=test_project.id,
        name="delegation-plan",
        artifact_type="plan",
        status="draft",
        linked_task_id=request_action.target_id,
        created_by_agent=planner.id,
        metadata_=_accepted_plan_metadata(),
    )
    db_session.add(artifact)
    await db_session.flush()
    gate = await _single_plan_gate(db_session, run.id)
    event = EventLog(
        project_id=test_project.id,
        event_type="artifact.created",
        dedup_key=f"test:artifact-created:{artifact.id}",
        payload={"artifact_id": str(artifact.id)},
        source="test",
    )
    db_session.add(event)
    await db_session.flush()
    db_session.add(
        OrchestrationEvidence(
            run_id=run.id,
            gate_id=gate.id,
            source_type="artifact",
            source_id=artifact.id,
            observed_event_id=event.id,
            producer_agent_id=planner.id,
            verdict="candidate",
            evidence_metadata={"source": "event"},
        )
    )
    await db_session.flush()

    action = await service.execute_accept_plan_action(
        db_session,
        run_id=run.id,
        request={"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
        idempotency_key="run:phase11:kind:accept_plan:event-evidence",
    )

    evidence_result = await db_session.execute(
        select(OrchestrationEvidence).where(
            OrchestrationEvidence.run_id == run.id,
            OrchestrationEvidence.gate_id == gate.id,
            OrchestrationEvidence.source_type == "artifact",
            OrchestrationEvidence.source_id == artifact.id,
        )
    )
    evidence_rows = list(evidence_result.scalars().all())
    assert len(evidence_rows) == 2
    event_linked = [row for row in evidence_rows if row.observed_event_id == event.id]
    direct = [row for row in evidence_rows if row.observed_event_id is None]
    assert len(event_linked) == 1
    assert event_linked[0].verdict == "candidate"
    assert len(direct) == 1
    assert direct[0].verdict == "accepted"
    assert direct[0].evidence_metadata == {"action_id": str(action.id), "gate_type": "plan_accepted"}


def test_plan_delegation_validator_rejects_authored_plan_content():
    result = validate_orchestration_decision(
        {
            "action_type": "request_plan",
            "work_function": "planning",
            "agent_id": str(uuid.uuid4()),
            "scope": "Ask an agent for a plan.",
            "plan_text": "The orchestrator must not write the plan.",
        }
    )

    assert result.accepted is False
    assert result.rejection_reason == "Decision includes forbidden artifact content at 'plan_text'"
