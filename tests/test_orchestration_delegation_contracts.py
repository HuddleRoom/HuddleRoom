import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql.functions import count

from huddleroom.models.agent import Agent
from huddleroom.models.event_log import EventLog
from huddleroom.models.orchestration import (
    OrchestrationAction, OrchestrationBudgetReservation, OrchestrationGoal, OrchestrationRun,
)
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
        "orchestration_contract": {
            **_contract(goal_id, run_id, action_id, request).model_dump(mode="json"),
            "expected_result": "work_report",
        },
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


@pytest.mark.parametrize(
    ("mode", "snapshot", "deliverable"),
    [
        ("roadmap", None, "Roadmap plan: a finite graph of distinct outcomes, defined progressively."),
        ("delivery", None, "Delivery plan: one bounded plan coordinating every work function this outcome needs."),
        ("implementation", None, "Implementation plan: a leaf technical plan for this outcome."),
        ("implementation", "roadmap", "Roadmap plan: a finite graph of distinct outcomes, defined progressively."),
    ],
)
def test_planning_contract_wording_by_mode(mode, snapshot, deliverable):
    from types import SimpleNamespace

    goal = SimpleNamespace(
        id=uuid.uuid4(), planning_mode=mode, objective="Ship it", orchestrator_context={},
        success_criteria=[], constraints={}, budget={}, goal_type="goal",
    )
    run = SimpleNamespace(id=uuid.uuid4(), planning_mode_snapshot=snapshot, plan_state={})
    action = SimpleNamespace(id=uuid.uuid4())
    contract = OrchestrationService()._planning_contract(
        goal, run, action, {"agent_id": str(uuid.uuid4()), "scope": "plan it"}
    )
    assert contract.deliverable == deliverable
    assert any(
        i.startswith("Optional specialist inputs") and "inputs to this plan, never co-authors" in i
        and "product strategist — sequencing" in i and "tester — qualification and evidence" in i
        and "UI/UX designer — interaction and accessibility" in i
        for i in contract.inputs
    )
    assert "You are the sole plan author: one plan that synthesizes any specialist inputs." in contract.success_evidence


def test_planning_contract_lists_eligible_parent_evidence_only_when_present():
    from types import SimpleNamespace

    artifact, evidence = uuid.uuid4(), uuid.uuid4()

    def contract(context):
        goal = SimpleNamespace(
            id=uuid.uuid4(), planning_mode="roadmap", objective="Ship it", orchestrator_context=context,
            success_criteria=[], constraints={}, budget={}, goal_type="roadmap",
        )
        run = SimpleNamespace(id=uuid.uuid4(), planning_mode_snapshot=None, plan_state={})
        return OrchestrationService()._planning_contract(
            goal, run, SimpleNamespace(id=uuid.uuid4()), {"agent_id": str(uuid.uuid4()), "scope": "plan it"}
        )

    nested = contract({"eligible_evidence": {"artifact_ids": [str(artifact)], "evidence_ids": [str(evidence)]}})
    assert nested.artifact_ids == [artifact]
    assert (
        "Eligible parent evidence (references only; re-verify before relying on them for any final criterion): "
        f"artifacts {artifact}; evidence {evidence}."
    ) in nested.inputs
    root = contract({})
    assert root.artifact_ids == [] and not any("Eligible parent evidence" in i for i in root.inputs)


def test_planning_contract_lists_retained_evidence_manifest_only_when_present():
    from types import SimpleNamespace

    gate, task, art, art2 = (uuid.uuid4() for _ in range(4))
    eligible = uuid.uuid4()

    def contract(manifest):
        goal = SimpleNamespace(
            id=uuid.uuid4(), planning_mode="roadmap", objective="Ship it",
            orchestrator_context={"eligible_evidence": {"artifact_ids": [str(art)]}},
            success_criteria=[], constraints={}, budget={}, goal_type="roadmap",
        )
        run = SimpleNamespace(
            id=uuid.uuid4(), planning_mode_snapshot=None,
            plan_state={"retained_evidence_manifest": manifest} if manifest else {},
        )
        return OrchestrationService()._planning_contract(
            goal, run, SimpleNamespace(id=uuid.uuid4()), {"agent_id": str(uuid.uuid4()), "scope": "plan it"}
        )

    with_manifest = contract([
        {"source_gate_id": str(gate), "source_task_id": str(task), "artifact_id": str(art), "revalidation_required": True},
        {"source_gate_id": str(eligible), "source_task_id": str(task), "artifact_id": str(art2), "revalidation_required": False},
        {"source_gate_id": str(eligible), "source_task_id": str(task), "artifact_id": None, "revalidation_required": False},
    ])
    assert with_manifest.artifact_ids == [art, art2]
    line = next(i for i in with_manifest.inputs if i.startswith("Retained evidence from superseded run"))
    assert line.startswith("Retained evidence from superseded run (re-verify; reuse only via retained gates):")
    assert f"{gate} | task {task} | artifact {art} | re-verify" in line
    assert f"{eligible} | task {task} | artifact {art2} | contract unchanged" in line
    assert f"artifact none | contract unchanged" in line
    plain = contract(None)
    assert plain.artifact_ids == [art] and not any("Retained evidence" in i for i in plain.inputs)


# --- orchestration_dependencies ---------------------------------------------

async def _dep_task(db_session, project_id, run, *, gate=None, status="done", run_id="same", meta_key="gate_id"):
    from huddleroom.models.orchestration import OrchestrationGate

    metadata = {}
    if gate is not False:
        gate_row = gate
        if gate_row is None:
            gate_row = OrchestrationGate(run_id=run.id, success_criterion_key="delegated", gate_type="evidence")
            db_session.add(gate_row)
            await db_session.flush()
        orch = {"run_id": str(run.id) if run_id == "same" else str(uuid.uuid4()), meta_key: str(gate_row.id)}
        metadata = {"orchestration": orch}
    elif run_id == "same":
        metadata = {"orchestration": {"run_id": str(run.id)}}
    task = Task(project_id=project_id, title="dep", status=status, metadata_=metadata)
    db_session.add(task)
    await db_session.flush()
    return task


@pytest.mark.asyncio
async def test_task_gate_id_resolution_order_and_wrong_run(db_session, test_project):
    from huddleroom.services.orchestration_dependencies import task_gate_id

    _, _, run = await _make_run(db_session, test_project.id)
    a, b, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    rid = str(run.id)
    both = Task(metadata_={
        "orchestration": {"run_id": rid, "gate_id": str(a), "plan_item_gate_id": str(b)},
        "orchestration_plan_item": {"plan_item_gate_id": str(c)},
    })
    assert task_gate_id(both, run.id) == a
    mid = Task(metadata_={
        "orchestration": {"run_id": rid, "gate_id": "bad", "plan_item_gate_id": str(b)},
        "orchestration_plan_item": {"plan_item_gate_id": str(c)},
    })
    assert task_gate_id(mid, run.id) == b
    last = Task(metadata_={"orchestration": {"run_id": rid}, "orchestration_plan_item": {"plan_item_gate_id": str(c)}})
    assert task_gate_id(last, run.id) == c
    assert task_gate_id(Task(metadata_={"orchestration": {"run_id": str(uuid.uuid4()), "gate_id": str(a)}}), run.id) is None


@pytest.mark.asyncio
async def test_dependency_satisfied_only_for_accepted_gate(db_session, test_project):
    from huddleroom.models.orchestration import OrchestrationGate
    from huddleroom.services.orchestration_dependencies import dependency_satisfied, first_unmet_dependency

    _, _, run = await _make_run(db_session, test_project.id)
    pid = test_project.id
    done_open = await _dep_task(db_session, pid, run, status="done")
    done_nogate = await _dep_task(db_session, pid, run, gate=False, status="done")
    failed = await _dep_task(db_session, pid, run, status="failed")
    cancelled = await _dep_task(db_session, pid, run, status="cancelled")
    accepted = await _dep_task(db_session, pid, run, status="done")
    gate = await db_session.get(OrchestrationGate, uuid.UUID(accepted.metadata_["orchestration"]["gate_id"]))
    gate.status = "accepted"
    await db_session.flush()

    for t in (done_open, done_nogate, failed, cancelled):
        assert not await dependency_satisfied(db_session, run, t.id)
    assert not await dependency_satisfied(db_session, run, uuid.uuid4())
    assert await dependency_satisfied(db_session, run, accepted.id)

    waiter = Task(project_id=pid, title="w", depends_on=[str(accepted.id), str(done_open.id)])
    assert await first_unmet_dependency(db_session, run, waiter) == (done_open.id, "open")
    waiter.depends_on = [str(done_nogate.id)]
    assert await first_unmet_dependency(db_session, run, waiter) == (done_nogate.id, "none")
    waiter.depends_on = [str(accepted.id)]
    assert await first_unmet_dependency(db_session, run, waiter) is None


@pytest.mark.asyncio
async def test_validate_dependencies_rejects_each_invalid_case(db_session, test_project):
    from huddleroom.services.orchestration_dependencies import validate_dependencies

    _, _, run = await _make_run(db_session, test_project.id)
    pid = test_project.id
    other = Project(name="Other", description="o", workspace_path="/tmp/other-dep", config={})
    db_session.add(other)
    await db_session.flush()

    good = await _dep_task(db_session, pid, run)
    await validate_dependencies(db_session, run, [good.id])
    await validate_dependencies(db_session, run, [])

    async def rejected(dep_ids, **kw):
        with pytest.raises(HTTPException) as exc_info:
            await validate_dependencies(db_session, run, dep_ids, **kw)
        assert exc_info.value.status_code == 409
        assert exc_info.value.detail.startswith("dependency_invalid:")

    await rejected([uuid.uuid4()])
    await rejected([(await _dep_task(db_session, other.id, run)).id])
    await rejected([(await _dep_task(db_session, pid, run, run_id="other")).id])
    await rejected([good.id], self_id=good.id)
    await rejected([(await _dep_task(db_session, pid, run, gate=False)).id])

    a = await _dep_task(db_session, pid, run)
    b = await _dep_task(db_session, pid, run)
    a.depends_on, b.depends_on = [str(b.id)], [str(a.id)]
    await db_session.flush()
    await rejected([a.id])
    new_id = uuid.uuid4()
    good.depends_on = [str(new_id)]
    await db_session.flush()
    await rejected([good.id], self_id=new_id)


# --- ad-hoc create_delegation_task.depends_on --------------------------------

async def _adhoc_dep_setup(db_session, project):
    agent = _agent("delegate", capabilities=["implementation"])
    db_session.add(agent)
    await db_session.flush()
    service, goal, run = await _make_run(db_session, project.id)
    return service, run, agent


async def _delegate(service, run, agent, key, *, db, **extra):
    return await service.execute_create_delegation_task_action(
        db, run_id=run.id, request={**_delegation_request(agent.id), **extra}, idempotency_key=f"run:t8b:{key}",
    )


@pytest.mark.asyncio
async def test_adhoc_delegation_stores_validated_dependency_everywhere(db_session, test_project):
    from huddleroom.schemas.task import TaskResponse

    service, run, agent = await _adhoc_dep_setup(db_session, test_project)
    dep = await _dep_task(db_session, test_project.id, run)

    action = await _delegate(service, run, agent, "ok", db=db_session, depends_on=[f" {dep.id} "])

    task = await db_session.get(Task, action.id)
    assert action.request["depends_on"] == [str(dep.id)]
    assert task.depends_on == [str(dep.id)]
    assert task.metadata_["orchestration_contract"]["depends_on"] == [str(dep.id)]
    assert TaskResponse.model_validate(task).depends_on == [dep.id]


@pytest.mark.asyncio
async def test_adhoc_delegation_without_dependency_keeps_legacy_request(db_session, test_project):
    service, run, agent = await _adhoc_dep_setup(db_session, test_project)

    action = await _delegate(service, run, agent, "none", db=db_session, depends_on=[])

    assert "depends_on" not in action.request
    assert action.request == _canonical_request(agent.id)
    assert (await db_session.get(Task, action.id)).depends_on is None


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["missing", "cross_project", "cross_run", "no_gate", "duplicate", "duplicate_case", "malformed", "not_list", "empty_string", "empty_object", "zero"])
async def test_adhoc_delegation_rejects_invalid_dependency_without_side_effects(db_session, test_project, case):
    from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision

    service, run, agent = await _adhoc_dep_setup(db_session, test_project)
    other = Project(name="Other", description="o", workspace_path="/tmp/other-dep-8b", config={})
    db_session.add(other)
    await db_session.flush()
    good = await _dep_task(db_session, test_project.id, run)
    deps = {
        "missing": [str(uuid.uuid4())],
        "cross_project": [str((await _dep_task(db_session, other.id, run)).id)],
        "cross_run": [str((await _dep_task(db_session, test_project.id, run, run_id="other")).id)],
        "no_gate": [str((await _dep_task(db_session, test_project.id, run, gate=False)).id)],
        "duplicate": [str(good.id), str(good.id)],
        "duplicate_case": [str(good.id), str(good.id).upper()],
        "malformed": ["not-a-uuid"],
        "not_list": str(good.id),
        "empty_string": "",
        "empty_object": {},
        "zero": 0,
    }[case]
    expected_detail = (
        "depends_on" if case in {"duplicate", "duplicate_case", "malformed", "not_list", "empty_string", "empty_object", "zero"}
        else "dependency_invalid:"
    )
    tasks_before = await _task_count(db_session, test_project.id)
    actions_before = await _action_count(db_session, run.id)
    decisions_before = await db_session.scalar(select(count(OrchestrationAuthorityDecision.id)))
    reservations_before = await db_session.scalar(select(count(OrchestrationBudgetReservation.id)))

    with pytest.raises(HTTPException) as exc_info:
        await _delegate(service, run, agent, case, db=db_session, depends_on=deps)

    assert 400 <= exc_info.value.status_code < 500
    assert expected_detail in exc_info.value.detail
    assert await _task_count(db_session, test_project.id) == tasks_before
    assert await _action_count(db_session, run.id) == actions_before
    assert await db_session.scalar(select(count(OrchestrationAuthorityDecision.id))) == decisions_before
    assert await db_session.scalar(select(count(OrchestrationBudgetReservation.id))) == reservations_before


@pytest.mark.asyncio
async def test_task_terminal_state_alone_does_not_satisfy_adhoc_dependency(db_session, test_project):
    from huddleroom.services.orchestration_dependencies import first_unmet_dependency

    service, run, agent = await _adhoc_dep_setup(db_session, test_project)
    dep = await _dep_task(db_session, test_project.id, run, status="done")
    action = await _delegate(service, run, agent, "gated", db=db_session, depends_on=[str(dep.id)])
    task = await db_session.get(Task, action.id)

    assert await first_unmet_dependency(db_session, run, task) == (dep.id, "open")


_PASSTHROUGH_AGENT_ID = str(uuid.uuid4())


def _passthrough_request(**fields):
    return {
        "action_type": "create_delegation_task", "agent_id": _PASSTHROUGH_AGENT_ID, "work_function": "investigation",
        "scope": "Audit.", "deliverable": "Findings.", **fields,
    }


def test_canonical_delegation_request_keeps_plan_item_fields_only_when_non_default():
    service = OrchestrationService()
    artifact_id = uuid.uuid4()
    legacy = service._canonical_delegation_task_request(_passthrough_request())

    assert service._canonical_delegation_task_request(
        _passthrough_request(priority=50, artifact_ids=[], why_runnable="  ")
    ) == legacy
    full = service._canonical_delegation_task_request(
        _passthrough_request(priority=0, artifact_ids=[str(artifact_id)], why_runnable=" ready ")
    )
    assert full == {**legacy, "priority": 0, "artifact_ids": [str(artifact_id)], "why_runnable": "ready"}


@pytest.mark.parametrize("fields", [
    {"priority": True}, {"priority": 101}, {"priority": -1}, {"priority": "80"}, {"priority": 1.5},
    {"artifact_ids": "x"}, {"artifact_ids": ["not-a-uuid"]}, {"why_runnable": 3},
])
def test_canonical_delegation_request_rejects_invalid_plan_item_fields_with_4xx(fields):
    with pytest.raises(HTTPException) as exc_info:
        OrchestrationService()._canonical_delegation_task_request(_passthrough_request(**fields))
    assert 400 <= exc_info.value.status_code < 500


@pytest.mark.asyncio
async def test_adhoc_optional_fields_flow_into_request_contract_and_task(db_session, test_project):
    from huddleroom.models.artifact import Artifact
    from huddleroom.schemas.task import TaskResponse

    service, run, agent = await _adhoc_dep_setup(db_session, test_project)
    artifact = Artifact(project_id=test_project.id, name="spec", artifact_type="document")
    db_session.add(artifact)
    await db_session.flush()
    scope = {"workspace_mutation": "staged"}

    action = await _delegate(
        service, run, agent, "opt", db=db_session, decision_use=" pick engine ", why_runnable=" no plan needed ",
        priority=80, artifact_ids=[f" {artifact.id} "], authorization_scope=scope,
    )

    task = await db_session.get(Task, action.id)
    contract = task.metadata_["orchestration_contract"]
    assert action.request["decision_use"] == "pick engine"
    assert action.request["why_runnable"] == "no plan needed"
    assert action.request["priority"] == 80 and action.request["artifact_ids"] == [str(artifact.id)]
    assert action.request["authorization_scope"]["workspace_mutation"] == "staged"
    assert contract["decision_use"] == "pick engine" and contract["priority"] == 80
    assert contract["authorization_scope"]["workspace_mutation"] == "staged"
    assert contract["artifact_ids"] == [str(artifact.id)]
    assert TaskResponse.model_validate(task).priority == 80


@pytest.mark.asyncio
async def test_adhoc_defaults_keep_legacy_request_and_unknown_artifact_creates_nothing(db_session, test_project):
    service, run, agent = await _adhoc_dep_setup(db_session, test_project)
    legacy = await _delegate(service, run, agent, "legacy", db=db_session, decision_use=" ", priority=50)
    assert legacy.request == _canonical_request(agent.id)

    before = await _task_count(db_session, test_project.id)
    with pytest.raises(HTTPException) as exc:
        await _delegate(service, run, agent, "bad-art", db=db_session, artifact_ids=[str(uuid.uuid4())])
    assert exc.value.status_code == 409 and exc.value.detail.startswith("artifact_invalid:")
    assert await _task_count(db_session, test_project.id) == before


@pytest.mark.asyncio
async def test_dispatch_unknown_artifact_is_failed_action_with_zero_tasks(db_session, test_project):
    from huddleroom.models.orchestration import OrchestrationDecision
    from huddleroom.services.orchestration_decision_dispatcher import OrchestrationDecisionDispatcher

    service, run, agent = await _adhoc_dep_setup(db_session, test_project)
    decision = OrchestrationDecision(
        run_id=run.id, decision_type="execution", validator_status="accepted", input_snapshot={},
        parsed_decision={**_delegation_request(agent.id), "artifact_ids": [str(uuid.uuid4())]},
    )
    db_session.add(decision)
    await db_session.flush()
    project_id = test_project.id
    before = await _task_count(db_session, project_id)

    action = await OrchestrationDecisionDispatcher(service).dispatch(db_session, run, decision)

    assert action.status == "failed" and "artifact_invalid" in (action.error or "")
    assert await _task_count(db_session, project_id) == before


@pytest.mark.parametrize("scope", ["wide", {"bogus": 1}])
def test_canonical_request_rejects_invalid_authorization_scope(scope):
    with pytest.raises(HTTPException) as exc:
        OrchestrationService()._canonical_delegation_task_request(
            {**_delegation_request(uuid.uuid4()), "authorization_scope": scope}
        )
    assert exc.value.status_code == 422


@pytest.mark.parametrize("scope", [{"external_writes": {"allowed": True}}, {"live_sends": {"allowed": True}}])
def test_canonical_request_rejects_self_granted_authorization_scope(scope):
    with pytest.raises(HTTPException) as exc:
        OrchestrationService()._canonical_delegation_task_request(
            {**_delegation_request(uuid.uuid4()), "authorization_scope": scope}
        )
    assert exc.value.status_code == 422 and "self-grant" in exc.value.detail


def test_canonical_request_accepts_null_optional_fields():
    request = OrchestrationService()._canonical_delegation_task_request({
        **_delegation_request(uuid.uuid4()), "decision_use": None, "authorization_scope": None, "priority": None,
        "artifact_ids": None, "why_runnable": None, "title": None, "depends_on": None,
    })
    assert "decision_use" not in request and "authorization_scope" not in request


# --- SPR 180/184: delegated remediation keeps the ordinary delegation contract --------------


async def _baseline_blocked_rows(db_session, project, monkeypatch):
    from tests.test_orchestration_recovery_delegation import executables, fail_process, seed, spy_build_command

    executables(monkeypatch)
    spy_build_command(monkeypatch)
    rows = await seed(db_session, project)
    failed = await fail_process(db_session, rows, "team_hierarchy")
    return rows, failed


@pytest.mark.asyncio
async def test_adhoc_executor_ignores_model_authored_admission_and_skip_flag(db_session, test_project, monkeypatch):
    rows, _failed = await _baseline_blocked_rows(db_session, test_project, monkeypatch)
    request = {
        **_delegation_request(rows.agent.id),
        "baseline_repair_admission": {"process_type": "team_hierarchy", "gate_id": str(uuid.uuid4()),
                                      "gate_fingerprint": "a" * 64},
        "skip_baseline_gate": True,
    }

    with pytest.raises(HTTPException) as exc:
        await rows.service.execute_create_delegation_task_action(
            db_session, run_id=rows.run.id, request=request, idempotency_key="run:t10:model-admission", ad_hoc=True,
        )

    assert exc.value.status_code == 409
    canonical = OrchestrationService()._canonical_delegation_task_request(request)
    assert "baseline_repair_admission" not in canonical and "skip_baseline_gate" not in canonical
    assert await _task_count(db_session, test_project.id) == 0


@pytest.mark.asyncio
async def test_forged_admission_context_without_approved_recovery_is_rejected(db_session, test_project, monkeypatch):
    from huddleroom.models.orchestration import OrchestrationDecision
    from huddleroom.schemas.orchestration_recovery import BaselineRepairAdmissionV1, RecoveryApplyDispatchContext
    from huddleroom.services.orchestration_decision_dispatcher import OrchestrationDecisionDispatcher

    rows, failed = await _baseline_blocked_rows(db_session, test_project, monkeypatch)
    decision = OrchestrationDecision(
        run_id=rows.run.id, decision_type="recovery_apply", validator_status="accepted", input_snapshot={},
        parsed_decision={**_delegation_request(rows.agent.id), "applies_decision_id": str(uuid.uuid4())},
    )
    db_session.add(decision)
    await db_session.flush()
    context = RecoveryApplyDispatchContext(
        purpose="apply", proposal_id=uuid.uuid4(), call_id="call-1", authority_decision_id=uuid.uuid4(),
        baseline_repair_admission=BaselineRepairAdmissionV1(
            process_type="team_hierarchy", gate_id=failed.id, gate_fingerprint="a" * 64),
    )

    action = await OrchestrationDecisionDispatcher(OrchestrationService()).dispatch(
        db_session, rows.run, decision, recovery_context=context)

    assert action is None or action.status == "failed"
    assert await _task_count(db_session, test_project.id) == 0


@pytest.mark.asyncio
async def test_approved_baseline_repair_reuses_adhoc_executor_without_broad_skip_flag(
    db_session, test_project, monkeypatch,
):
    from tests.test_orchestration_recovery_delegation import HIERARCHY_RETRY, approve, execute, propose

    rows, _failed = await _baseline_blocked_rows(db_session, test_project, monkeypatch)
    calls = []
    original = OrchestrationService.execute_create_delegation_task_action

    async def spy(self, db, run_id, request, idempotency_key, decision_id=None, **kwargs):
        calls.append({"request": request, "kwargs": kwargs})
        return await original(self, db, run_id, request, idempotency_key, decision_id, **kwargs)

    monkeypatch.setattr(OrchestrationService, "execute_create_delegation_task_action", spy)
    approved = await approve(db_session, rows, await propose(db_session, rows, HIERARCHY_RETRY))

    receipt = await execute(db_session, approved)

    assert receipt.status == "waiting"
    assert len(calls) == 1
    assert calls[0]["kwargs"].get("skip_baseline_gate", False) is False
    assert calls[0]["kwargs"].get("ad_hoc") is True
    assert "baseline_repair_admission" not in calls[0]["request"]
    task = await db_session.get(Task, receipt.action_id)
    assert task.metadata_["orchestration_contract"]["agent_id"] == str(rows.agent.id)
    assert task.metadata_["orchestration"]["action_id"] == str(receipt.action_id)
