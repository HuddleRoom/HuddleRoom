import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql.functions import count

from huddleroom.models.agent import Agent
from huddleroom.models.event_log import EventLog
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGoal, OrchestrationRun
from huddleroom.models.project import Project
from huddleroom.models.task import Task
from huddleroom.schemas.orchestration import OrchestrationDelegationContract, OrchestrationGoalCreate
from huddleroom.schemas.task import TaskCreate
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
            objective="Ship delegated work",
            success_criteria=[{"key": "delegated", "description": "Delegated work is tracked by task contract."}],
            constraints={"owned_files": ["huddleroom/services/orchestration_service.py"]},
            budget={"caps": {"max_tokens": 20001}},
        ),
        created_by_user_id=None,
    )
    await _complete_goal_definition(db_session, goal, run)
    goal.status = "active"
    run.status = "running"
    run.phase = "authorized"
    await db_session.flush()
    return service, goal, run


async def _event_rows(db_session, project_id, event_type):
    result = await db_session.execute(
        select(EventLog)
        .where(EventLog.project_id == project_id, EventLog.event_type == event_type)
        .order_by(EventLog.seq.asc())
    )
    return list(result.scalars().all())


async def _task_count(db_session, project_id):
    return await db_session.scalar(select(count(Task.id)).where(Task.project_id == project_id))


async def _action_count(db_session, run_id):
    return await db_session.scalar(select(count(OrchestrationAction.id)).where(OrchestrationAction.run_id == run_id))


async def _cleanup_replay_rows(session_factory, project_id, agent_id):
    async with session_factory() as cleanup_db:
        await cleanup_db.execute(delete(Task).where(Task.project_id == project_id))
        await cleanup_db.execute(
            delete(OrchestrationAction).where(
                OrchestrationAction.run_id.in_(
                    select(OrchestrationRun.id)
                    .join(OrchestrationGoal, OrchestrationGoal.id == OrchestrationRun.goal_id)
                    .where(OrchestrationGoal.project_id == project_id)
                )
            )
        )
        await cleanup_db.execute(
            delete(OrchestrationRun).where(
                OrchestrationRun.goal_id.in_(
                    select(OrchestrationGoal.id).where(OrchestrationGoal.project_id == project_id)
                )
            )
        )
        await cleanup_db.execute(delete(OrchestrationGoal).where(OrchestrationGoal.project_id == project_id))
        await cleanup_db.execute(delete(Agent).where(Agent.id == agent_id))
        await cleanup_db.execute(delete(Project).where(Project.id == project_id))
        await cleanup_db.commit()


def _agent(name_prefix: str, role: str = "developer", capabilities: list[str] | None = None, *, is_active: bool = True):
    return Agent(
        name=f"{name_prefix}-{uuid.uuid4()}",
        role=role,
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=capabilities or [],
        config={},
        is_active=is_active,
    )


def _delegation_request(agent_id: uuid.UUID, parent_task_id: uuid.UUID | None = None):
    request = {
        "action_type": "create_delegation_task",
        "agent_id": f" {agent_id} ",
        "work_function": " implementation ",
        "scope": " Implement the accepted plan item without changing unrelated files. ",
        "inputs": [" ROADMAP.md delegation contracts ", " ", "tests/test_orchestration_delegation_contracts.py"],
        "deliverable": " A code change plus focused pytest output. ",
        "forbidden_work": [" Do not edit dashboard files. ", ""],
        "success_evidence": [" pytest output for tests/test_orchestration_delegation_contracts.py "],
        "budget": {"max_tokens": 20000},
        "report_schema": {"changed_files": "list[str]", "tests": "list[str]"},
    }
    if parent_task_id is not None:
        request["parent_task_id"] = f" {parent_task_id} "
    return request


def _canonical_request(agent_id: uuid.UUID, parent_task_id: uuid.UUID | None = None):
    return {
        "action_type": "create_delegation_task",
        "agent_id": str(agent_id),
        "work_function": "implementation",
        "scope": "Implement the accepted plan item without changing unrelated files.",
        "inputs": ["ROADMAP.md delegation contracts", "tests/test_orchestration_delegation_contracts.py"],
        "deliverable": "A code change plus focused pytest output.",
        "forbidden_work": ["Do not edit dashboard files."],
        "success_evidence": ["pytest output for tests/test_orchestration_delegation_contracts.py"],
        "budget": {"max_tokens": 20000},
        "report_schema": {"changed_files": "list[str]", "tests": "list[str]"},
        "parent_task_id": str(parent_task_id) if parent_task_id is not None else None,
        "source_session_id": None,
    }


def _contract(goal_id: uuid.UUID, run_id: uuid.UUID, action_id: uuid.UUID, request: dict):
    return OrchestrationDelegationContract(
        goal_id=goal_id,
        run_id=run_id,
        action_id=action_id,
        agent_id=uuid.UUID(request["agent_id"]),
        work_function=request["work_function"],
        scope=request["scope"],
        inputs=request["inputs"],
        deliverable=request["deliverable"],
        forbidden_work=request["forbidden_work"],
        success_evidence=request["success_evidence"],
        budget=request["budget"],
        report_schema=request["report_schema"],
        parent_task_id=uuid.UUID(request["parent_task_id"]) if request["parent_task_id"] is not None else None,
    )


def _metadata(goal_id: uuid.UUID, run_id: uuid.UUID, action_id: uuid.UUID, request: dict):
    return {
        "orchestration": {
            "goal_id": str(goal_id),
            "run_id": str(run_id),
            "action_id": str(action_id),
            "work_function": request["work_function"],
        },
        "orchestration_contract": _contract(goal_id, run_id, action_id, request).model_dump(mode="json"),
    }


def test_delegation_contract_serializes_ids_for_task_metadata():
    goal_id = uuid.uuid4()
    run_id = uuid.uuid4()
    action_id = uuid.uuid4()
    agent_id = uuid.uuid4()
    parent_task_id = uuid.uuid4()

    contract = OrchestrationDelegationContract(
        goal_id=goal_id,
        run_id=run_id,
        action_id=action_id,
        agent_id=agent_id,
        work_function="implementation",
        scope="Implement the accepted plan item.",
        inputs=["ROADMAP.md delegation contracts"],
        deliverable="A code change plus focused pytest output.",
        forbidden_work=["Do not edit dashboard files."],
        success_evidence=["pytest output"],
        budget={"max_tokens": 20000},
        report_schema={"changed_files": "list[str]", "tests": "list[str]"},
        parent_task_id=parent_task_id,
    )

    dumped = contract.model_dump(mode="json")

    assert dumped == {
        "goal_id": str(goal_id),
        "run_id": str(run_id),
        "action_id": str(action_id),
        "agent_id": str(agent_id),
        "work_function": "implementation",
        "scope": "Implement the accepted plan item.",
        "inputs": ["ROADMAP.md delegation contracts"],
        "deliverable": "A code change plus focused pytest output.",
        "forbidden_work": ["Do not edit dashboard files."],
        "success_evidence": ["pytest output"],
        "budget": {"max_tokens": 20000},
        "report_schema": {"changed_files": "list[str]", "tests": "list[str]"},
        "parent_task_id": str(parent_task_id),
    }


def test_delegation_task_title_title_cases_and_caps_length():
    title = OrchestrationService._delegation_task_title(
        "implementation_review",
        "ship a very long deliverable description that should be truncated before it sprawls",
    )

    assert title.startswith("Implementation Review:")
    assert len(title) <= 80
    assert title.endswith("...")


@pytest.mark.asyncio
async def test_task_service_can_create_task_with_reserved_id(db_session, test_project):
    task_id = uuid.uuid4()

    task = await TaskService().create(
        db_session,
        test_project.id,
        TaskCreate(title="Delegated work", metadata={"source": "delegation-contract"}),
        task_id=task_id,
    )

    assert task.id == task_id
    assert task.project_id == test_project.id
    assert task.status == "backlog"
    assert task.metadata_ == {"source": "delegation-contract"}

    events = await _event_rows(db_session, test_project.id, "task.created")
    assert len(events) == 1
    assert events[0].payload["task_id"] == str(task_id)


@pytest.mark.asyncio
async def test_execute_create_delegation_task_action_creates_task_with_contract(db_session, test_project):
    agent = _agent("delegate", capabilities=["implementation"])
    db_session.add(agent)
    await db_session.flush()
    service, goal, run = await _make_run(db_session, test_project.id)

    action = await service.execute_create_delegation_task_action(
        db_session,
        run_id=run.id,
        request=_delegation_request(agent.id),
        idempotency_key="run:phase10:kind:create_delegation_task:implementation",
    )

    assert action.status == "completed"
    assert action.action_type == "create_delegation_task"
    assert action.target_type == "task"
    assert action.target_id == action.id
    assert action.request == _canonical_request(agent.id)

    task = await db_session.get(Task, action.id)
    assert task is not None
    assert task.title == "Implementation: A code change plus focused pytest output."
    assert task.description is not None
    assert goal.objective in task.description
    assert task.assigned_to == agent.id
    assert task.parent_id is None
    assert task.metadata_ == _metadata(goal.id, run.id, action.id, action.request)

    events = await _event_rows(db_session, test_project.id, "orchestration.delegation_task_created")
    assert len(events) == 1
    assert events[0].payload == {
        "goal_id": str(goal.id),
        "run_id": str(run.id),
        "action_id": str(action.id),
        "task_id": str(task.id),
        "agent_id": str(agent.id),
        "work_function": "implementation",
    }


@pytest.mark.asyncio
async def test_execute_create_delegation_task_action_replay_reuses_original_task_and_contract(
    db_session, test_project
):
    agent = _agent("delegate", capabilities=["implementation"])
    other_agent = _agent("other", capabilities=["implementation"])
    db_session.add_all([agent, other_agent])
    await db_session.flush()
    service, goal, run = await _make_run(db_session, test_project.id)

    first = await service.execute_create_delegation_task_action(
        db_session,
        run_id=run.id,
        request=_delegation_request(agent.id),
        idempotency_key="run:phase10:kind:create_delegation_task:replay",
    )
    second = await service.execute_create_delegation_task_action(
        db_session,
        run_id=run.id,
        request=_delegation_request(other_agent.id),
        idempotency_key="run:phase10:kind:create_delegation_task:replay",
    )

    assert second.id == first.id
    assert second.status == "completed"
    assert second.request == _canonical_request(agent.id)

    task_count = await _task_count(db_session, test_project.id)
    assert task_count == 1
    task = await db_session.get(Task, first.id)
    assert task.metadata_ == _metadata(goal.id, run.id, first.id, first.request)
    assert task.assigned_to == agent.id

    events = await _event_rows(db_session, test_project.id, "orchestration.delegation_task_created")
    assert len(events) == 1


@pytest.mark.asyncio
async def test_execute_create_delegation_task_action_partial_replay_reuses_existing_task(
    db_session, test_project
):
    agent = _agent("delegate", capabilities=["implementation"])
    db_session.add(agent)
    await db_session.flush()
    service, goal, run = await _make_run(db_session, test_project.id)
    request = _canonical_request(agent.id)
    action = await service.reserve_action(
        db_session,
        run_id=run.id,
        idempotency_key="run:phase10:kind:create_delegation_task:partial",
        action_type="create_delegation_task",
        request=request,
    )
    existing_task = Task(
        id=action.id,
        project_id=test_project.id,
        title="Recovered delegated task",
        assigned_to=agent.id,
        metadata_=_metadata(goal.id, run.id, action.id, request),
    )
    db_session.add(existing_task)
    await db_session.flush()

    completed = await service.execute_create_delegation_task_action(
        db_session,
        run_id=run.id,
        request=_delegation_request(agent.id),
        idempotency_key="run:phase10:kind:create_delegation_task:partial",
    )

    assert completed.id == action.id
    assert completed.status == "completed"
    assert completed.target_type == "task"
    assert completed.target_id == existing_task.id

    task_count = await _task_count(db_session, test_project.id)
    assert task_count == 1
    task = await db_session.get(Task, action.id)
    assert task.metadata_ == _metadata(goal.id, run.id, action.id, request)

    events = await _event_rows(db_session, test_project.id, "orchestration.delegation_task_created")
    assert len(events) == 1


@pytest.mark.asyncio
async def test_find_task_for_action_matches_metadata_without_project_scan(db_session, test_project, monkeypatch):
    agent = _agent("delegate", capabilities=["implementation"])
    db_session.add(agent)
    await db_session.flush()
    service, goal, run = await _make_run(db_session, test_project.id)
    request = _canonical_request(agent.id)
    action = await service.reserve_action(
        db_session,
        run_id=run.id,
        idempotency_key="run:phase10:kind:create_delegation_task:metadata-fallback",
        action_type="create_delegation_task",
        request=request,
    )
    task = Task(
        id=uuid.uuid4(),
        project_id=test_project.id,
        title="Recovered delegated task",
        assigned_to=agent.id,
        metadata_=_metadata(goal.id, run.id, action.id, request),
    )
    db_session.add(task)
    await db_session.flush()

    async def fail_get(*_args, **_kwargs):
        return None

    monkeypatch.setattr(TaskService, "get", fail_get)

    found = await service._find_task_for_action(db_session, test_project.id, action, run_id=run.id)

    assert found is not None
    assert found.id == task.id


@pytest.mark.asyncio
@pytest.mark.parametrize("precondition_type", ["inactive_agent", "missing_parent"])
async def test_execute_create_delegation_task_action_replay_of_reserved_action_with_terminal_failure(
    test_engine, tmp_path, precondition_type,
):
    session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    project_id = None
    agent_id = None
    try:
        async with session_factory() as setup_db:
            project = Project(
                name="Replay Project",
                description="Committed replay setup",
                workspace_path=str(tmp_path),
                config={},
            )
            # Create baseline agent with capabilities needed by team_hierarchy
            baseline_agent = _agent("baseline", capabilities=["review", "validation"])
            setup_db.add_all([project, baseline_agent])
            await setup_db.flush()
            service, _, run = await _make_run(setup_db, project.id)

            # For inactive_agent test, create the delegation agent AFTER baseline completes
            # so it's not part of team_hierarchy fingerprint. This prevents fingerprint staling
            # when the agent is later deactivated.
            if precondition_type == "inactive_agent":
                agent = _agent("delegate", capabilities=["implementation"])
                setup_db.add(agent)
                await setup_db.flush()
            else:
                # For missing_parent, use the baseline agent for consistency with original test
                agent = baseline_agent

            parent = None
            if precondition_type == "missing_parent":
                parent = await TaskService().create(
                    setup_db,
                    project.id,
                    TaskCreate(title="Parent"),
                )

            parent_arg = parent.id if parent else None
            action = await service.reserve_action(
                setup_db,
                run_id=run.id,
                idempotency_key=f"run:phase10:kind:create_delegation_task:reserved-{precondition_type}",
                action_type="create_delegation_task",
                request=_canonical_request(agent.id, parent_arg),
            )
            await setup_db.commit()
            project_id = project.id
            agent_id = agent.id
            run_id = run.id
            action_id = action.id
            parent_id = parent.id if parent else None

        if precondition_type == "inactive_agent":
            async with session_factory() as mutate_db:
                agent = await mutate_db.get(Agent, agent_id)
                agent.is_active = False
                await mutate_db.commit()
            error_message = "Agent not found"
        else:  # missing_parent
            async with session_factory() as mutate_db:
                parent = await mutate_db.get(Task, parent_id)
                await mutate_db.delete(parent)
                await mutate_db.commit()
            error_message = "Parent task not found"

        async with session_factory() as replay_db:
            service = OrchestrationService()
            with pytest.raises(HTTPException) as exc_info:
                await service.execute_create_delegation_task_action(
                    replay_db,
                    run_id=run_id,
                    request=_delegation_request(agent_id, parent_id),
                    idempotency_key=f"run:phase10:kind:create_delegation_task:reserved-{precondition_type}",
                )

            assert exc_info.value.status_code == 404

        async with session_factory() as verify_db:
            action = await verify_db.get(OrchestrationAction, action_id)
            assert action.status == "failed"
            assert action.error == error_message
            assert await _task_count(verify_db, project_id) == 0
    finally:
        if project_id is not None and agent_id is not None:
            await _cleanup_replay_rows(session_factory, project_id, agent_id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_case,expected_status,expected_detail,request_builder,agent_setup",
    [
        ("malformed", 400, None,
         lambda agent_id: {"action_type": "create_delegation_task", "agent_id": str(agent_id)},
         lambda db, agent: None),
        ("missing_agent", 404, "Agent not found",
         lambda agent_id: _delegation_request(uuid.uuid4()),
         lambda db, agent: None),
        ("inactive_agent", 404, "Agent not found",
         lambda agent_id: _delegation_request(agent_id),
         lambda db, agent: setattr(agent, "is_active", False)),
    ],
)
async def test_execute_create_delegation_task_action_pre_validation_failures(
    db_session, test_project, error_case, expected_status, expected_detail, request_builder, agent_setup
):
    agent = _agent("delegate", capabilities=["implementation"])
    if error_case == "inactive_agent":
        agent.is_active = False
    db_session.add(agent)
    await db_session.flush()
    service, _, run = await _make_run(db_session, test_project.id)

    request = request_builder(agent.id)
    event_count_before = None
    if error_case == "malformed":
        event_count_before = await db_session.scalar(
            select(count(EventLog.id)).where(EventLog.project_id == test_project.id)
        )

    with pytest.raises(HTTPException) as exc_info:
        await service.execute_create_delegation_task_action(
            db_session,
            run_id=run.id,
            request=request,
            idempotency_key=f"run:phase10:kind:create_delegation_task:{error_case}",
        )

    assert exc_info.value.status_code == expected_status
    if expected_detail:
        assert exc_info.value.detail == expected_detail

    action = (
        await db_session.execute(
            select(OrchestrationAction).where(
                OrchestrationAction.idempotency_key == f"run:phase10:kind:create_delegation_task:{error_case}"
            )
        )
    ).scalar_one_or_none()
    assert action is None
    assert await _task_count(db_session, test_project.id) == 0

    if error_case == "malformed":
        assert await db_session.scalar(select(count(EventLog.id)).where(EventLog.project_id == test_project.id)) == event_count_before


@pytest.mark.asyncio
async def test_execute_create_delegation_task_action_checks_run_status_before_agent_validation(
    db_session, test_project
):
    service, _, run = await _make_run(db_session, test_project.id)
    run.status = "completed"
    await db_session.flush()

    with pytest.raises(HTTPException) as exc_info:
        await service.execute_create_delegation_task_action(
            db_session,
            run_id=run.id,
            request=_delegation_request(uuid.uuid4()),
            idempotency_key="run:phase10:kind:create_delegation_task:paused-missing-agent",
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Orchestration run is completed"
    assert await _action_count(db_session, run.id) == 0


@pytest.mark.asyncio
async def test_execute_create_delegation_task_action_missing_parent_fails_before_reserving(
    db_session, test_project
):
    agent = _agent("delegate", capabilities=["implementation"])
    db_session.add(agent)
    await db_session.flush()
    service, _, run = await _make_run(db_session, test_project.id)
    parent_id = uuid.uuid4()

    with pytest.raises(HTTPException) as exc_info:
        await service.execute_create_delegation_task_action(
            db_session,
            run_id=run.id,
            request=_delegation_request(agent.id, parent_id),
            idempotency_key="run:phase10:kind:create_delegation_task:missing-parent",
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Parent task not found"
    action = (
        await db_session.execute(
            select(OrchestrationAction).where(
                OrchestrationAction.idempotency_key == "run:phase10:kind:create_delegation_task:missing-parent"
            )
        )
    ).scalar_one_or_none()
    assert action is None
    assert await _task_count(db_session, test_project.id) == 0
