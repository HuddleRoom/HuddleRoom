import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from huddleroom.models.agent import Agent
from huddleroom.models.artifact import Artifact
from huddleroom.models.event_log import EventLog
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGate
from huddleroom.models.task import Task
from huddleroom.schemas.orchestration import OrchestrationGoalCreate, OrchestrationPlanItem
from huddleroom.services.orchestration_service import OrchestrationService


def _agent(
    name_prefix: str,
    role: str = "developer",
    capabilities: list[str] | None = None,
    *,
    is_active: bool = True,
) -> Agent:
    return Agent(
        name=f"{name_prefix}-{uuid.uuid4()}",
        role=role,
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=capabilities or ["implementation"],
        config={},
        is_active=is_active,
    )


def _raw_plan_item(
    agent_id: uuid.UUID | None = None,
    item_id: str = " implement-api ",
    work_function: str = " implementation ",
) -> dict:
    item = {
        "id": item_id,
        "title": " Implement API endpoint ",
        "work_function": work_function,
        "scope": " Implement the accepted plan item without changing unrelated files. ",
        "deliverable": " A code change plus focused pytest output. ",
        "inputs": [" ROADMAP.md plan-to-work expansion ", " "],
        "forbidden_work": [" Do not edit dashboard files. ", ""],
        "success_evidence": [" pytest output for tests/test_orchestration_plan_to_work_expansion.py "],
        "required_capabilities": [" implementation ", " "],
        "required_evidence": {"required_source_types": ["task"], "min_count": 1},
    }
    if agent_id is not None:
        item["agent_id"] = f" {agent_id} "
    return item


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
            objective="Ship plan-to-work expansion",
            success_criteria=[{"key": "expanded", "description": "Accepted plan items become delegated work."}],
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


async def _run_rows(db_session, model, run_id, **filters):
    query = select(model).where(model.run_id == run_id)
    for field_name, value in filters.items():
        query = query.where(getattr(model, field_name) == value)
    result = await db_session.execute(query)
    return list(result.scalars().all())


async def _make_accepted_plan(db_session, project_id, planner: Agent, plan_items: list[dict]):
    service, goal, run = await _make_run(db_session, project_id)
    request_action = await service.execute_request_plan_action(
        db_session,
        run_id=run.id,
        request={
            "action_type": "request_plan",
            "agent_id": str(planner.id),
            "work_function": "planning",
            "scope": "Create a plan with independently verifiable items.",
        },
        idempotency_key="run:phase12:kind:request_plan",
    )
    artifact = Artifact(
        project_id=project_id,
        name="expansion-plan",
        artifact_type="plan",
        status="draft",
        linked_task_id=request_action.target_id,
        created_by_agent=planner.id,
        metadata_={"kind": "implementation_plan", "plan_items": plan_items},
    )
    db_session.add(artifact)
    await db_session.flush()
    await service.execute_accept_plan_action(
        db_session,
        run_id=run.id,
        request={"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
        idempotency_key="run:phase12:kind:accept_plan",
    )
    return service, goal, run, artifact


def test_plan_item_schema_normalizes_agent_plan_item():
    agent_id = uuid.uuid4()

    item = OrchestrationPlanItem.model_validate(_raw_plan_item(agent_id))

    assert item.id == "implement-api"
    assert item.title == "Implement API endpoint"
    assert item.work_function == "implementation"
    assert item.scope == "Implement the accepted plan item without changing unrelated files."
    assert item.deliverable == "A code change plus focused pytest output."
    assert item.agent_id == agent_id
    assert item.inputs == ["ROADMAP.md plan-to-work expansion"]
    assert item.forbidden_work == ["Do not edit dashboard files."]
    assert item.success_evidence == ["pytest output for tests/test_orchestration_plan_to_work_expansion.py"]
    assert item.required_capabilities == ["implementation"]
    assert item.required_evidence == {"required_source_types": ["task"], "min_count": 1}


def test_plan_items_from_artifact_rejects_missing_metadata():
    artifact = Artifact(
        project_id=uuid.uuid4(),
        name="accepted-plan",
        artifact_type="plan",
        metadata_={"kind": "implementation_plan"},
    )

    with pytest.raises(HTTPException) as exc_info:
        OrchestrationService()._plan_items_from_artifact(artifact)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Accepted plan artifact must include metadata.plan_items"


def test_plan_items_from_artifact_rejects_duplicate_ids():
    artifact = Artifact(
        project_id=uuid.uuid4(),
        name="accepted-plan",
        artifact_type="plan",
        metadata_={"plan_items": [_raw_plan_item(), {**_raw_plan_item(), "title": "Second item"}]},
    )

    with pytest.raises(HTTPException) as exc_info:
        OrchestrationService()._plan_items_from_artifact(artifact)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Duplicate plan item id 'implement-api'"


@pytest.mark.asyncio
async def test_execute_expand_plan_item_action_creates_gate_and_work_task(db_session, test_project):
    planner = _agent("planner", role="planner", capabilities=["planning"])
    developer = _agent("developer", role="developer", capabilities=["implementation"])
    db_session.add_all([planner, developer])
    await db_session.flush()
    service, goal, run, artifact = await _make_accepted_plan(
        db_session,
        test_project.id,
        planner,
        [_raw_plan_item(developer.id)],
    )

    action = await service.execute_expand_plan_item_action(
        db_session,
        run_id=run.id,
        request={
            "action_type": "expand_plan_item",
            "plan_item_id": " implement-api ",
            "work_function": " implementation ",
        },
        idempotency_key=f"run:{run.id}:kind:expand_plan_item:plan_item:implement-api",
    )

    assert action.status == "completed"
    assert action.action_type == "expand_plan_item"
    assert action.request == {
        "action_type": "expand_plan_item",
        "plan_item_id": "implement-api",
        "work_function": "implementation",
    }
    assert action.target_type == "task"
    assert action.target_id is not None

    task = await db_session.get(Task, action.target_id)
    assert task is not None
    assert task.assigned_to == developer.id
    assert task.status == "backlog"
    assert task.metadata_["orchestration"]["goal_id"] == str(goal.id)
    assert task.metadata_["orchestration"]["run_id"] == str(run.id)
    assert task.metadata_["orchestration"]["work_function"] == "implementation"
    assert task.metadata_["orchestration"]["plan_item_id"] == "implement-api"
    assert task.metadata_["orchestration"]["expand_action_id"] == str(action.id)
    assert task.metadata_["orchestration_plan_item"]["id"] == "implement-api"
    assert task.metadata_["orchestration_plan_item"]["accepted_plan_artifact_id"] == str(artifact.id)

    gate_result = await db_session.execute(
        select(OrchestrationGate).where(
            OrchestrationGate.run_id == run.id,
            OrchestrationGate.success_criterion_key == "plan_item:implement-api",
            OrchestrationGate.gate_type == "work_completed",
        )
    )
    gate = gate_result.scalar_one()
    assert gate.status == "open"
    assert gate.required_evidence == {
        "required_source_types": ["task", "verification"],
        "min_count": 2,
        "requires_independent_agent": True,
        "plan_item_id": "implement-api",
        "success_criterion_keys": ["expanded"],
    }
    assert task.metadata_["orchestration"]["plan_item_gate_id"] == str(gate.id)

    delegation_result = await db_session.execute(
        select(OrchestrationAction).where(
            OrchestrationAction.run_id == run.id,
            OrchestrationAction.action_type == "create_delegation_task",
            OrchestrationAction.target_id == task.id,
        )
    )
    delegation_action = delegation_result.scalar_one()
    assert delegation_action.status == "completed"
    assert delegation_action.request["scope"] == "Implement the accepted plan item without changing unrelated files."
    assert delegation_action.request["inputs"] == [
        f"Accepted plan artifact: {artifact.id}",
        "Plan item: implement-api",
        "ROADMAP.md plan-to-work expansion",
    ]

    assert run.plan_state["expanded_items"] == [
        {
            "plan_item_id": "implement-api",
            "work_function": "implementation",
            "expand_action_id": str(action.id),
            "delegation_action_id": str(delegation_action.id),
            "task_id": str(task.id),
            "gate_id": str(gate.id),
        }
    ]

    events = await _event_rows(db_session, test_project.id, "orchestration.plan_item_expanded")
    assert len(events) == 1
    assert events[0].payload == {
        "goal_id": str(goal.id),
        "run_id": str(run.id),
        "action_id": str(action.id),
        "plan_item_id": "implement-api",
        "work_function": "implementation",
        "delegation_action_id": str(delegation_action.id),
        "task_id": str(task.id),
        "gate_id": str(gate.id),
    }


@pytest.mark.asyncio
async def test_release_backfills_sole_criterion_links_on_legacy_expanded_work_idempotently(
    db_session, test_engine, test_project
):
    """A committed pre-link one-criterion plan is repaired once across restarts."""
    planner = _agent("legacy-planner", role="planner", capabilities=["planning"])
    developer = _agent("legacy-developer", capabilities=["implementation"])
    db_session.add_all([planner, developer])
    await db_session.flush()
    service, goal, run, _artifact = await _make_accepted_plan(
        db_session, test_project.id, planner, [_raw_plan_item(developer.id)]
    )
    action = await service.execute_expand_plan_item_action(
        db_session,
        run.id,
        {"action_type": "expand_plan_item", "plan_item_id": "implement-api", "work_function": "implementation"},
        f"run:{run.id}:kind:expand_plan_item:plan_item:implement-api",
    )
    task = await db_session.get(Task, action.target_id)
    gate = await db_session.scalar(
        select(OrchestrationGate).where(
            OrchestrationGate.run_id == run.id,
            OrchestrationGate.success_criterion_key == "plan_item:implement-api",
        )
    )
    assert task is not None and gate is not None
    gate.required_evidence = {
        key: value for key, value in gate.required_evidence.items()
        if key != "success_criterion_keys"
    }
    task.metadata_ = {
        **task.metadata_,
        "orchestration": {
            key: value for key, value in task.metadata_["orchestration"].items()
            if key != "success_criterion_keys"
        },
        "orchestration_plan_item": {
            key: value for key, value in task.metadata_["orchestration_plan_item"].items()
            if key != "success_criterion_keys"
        },
    }
    run.phase = "authorized"
    await db_session.commit()

    sessions = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with sessions.begin() as reconcile_db:
        persisted_goal = await reconcile_db.get(type(goal), goal.id)
        persisted_run = await reconcile_db.get(type(run), run.id)
        persisted_gate = await reconcile_db.get(type(gate), gate.id)
        persisted_task = await reconcile_db.get(Task, task.id)
        assert persisted_goal is not None and persisted_run is not None
        assert persisted_gate is not None and persisted_task is not None
        assert "success_criterion_keys" not in persisted_gate.required_evidence
        assert "success_criterion_keys" not in persisted_task.metadata_["orchestration"]
        await OrchestrationService()._release_ready_work(reconcile_db, persisted_goal, persisted_run)

    async with sessions.begin() as replay_db:
        persisted_goal = await replay_db.get(type(goal), goal.id)
        persisted_run = await replay_db.get(type(run), run.id)
        persisted_gate = await replay_db.get(type(gate), gate.id)
        persisted_task = await replay_db.get(Task, task.id)
        assert persisted_goal is not None and persisted_run is not None
        assert persisted_gate is not None and persisted_task is not None
        first_required = dict(persisted_gate.required_evidence)
        first_metadata = dict(persisted_task.metadata_)
        assert first_required["success_criterion_keys"] == ["expanded"]
        assert first_metadata["orchestration"]["success_criterion_keys"] == ["expanded"]
        assert first_metadata["orchestration_plan_item"]["success_criterion_keys"] == ["expanded"]
        assert await OrchestrationService()._release_ready_work(replay_db, persisted_goal, persisted_run) == 0
        assert persisted_gate.required_evidence == first_required
        assert persisted_task.metadata_ == first_metadata


@pytest.mark.asyncio
async def test_expand_plan_item_replays_do_not_duplicate_or_rewrite_state(db_session, test_project):
    planner = _agent("planner", role="planner", capabilities=["planning"])
    developer = _agent("developer", role="developer", capabilities=["implementation"])
    db_session.add_all([planner, developer])
    await db_session.flush()
    service, _goal, run, _artifact = await _make_accepted_plan(
        db_session,
        test_project.id,
        planner,
        [_raw_plan_item(developer.id)],
    )
    project_id = test_project.id
    run_id = run.id
    request = {
        "action_type": "expand_plan_item",
        "plan_item_id": " implement-api ",
        "work_function": " implementation ",
    }

    first = await service.execute_expand_plan_item_action(
        db_session,
        run_id=run_id,
        request=request,
        idempotency_key=f"run:{run_id}:kind:expand_plan_item:plan_item:implement-api",
    )
    same_key = await service.execute_expand_plan_item_action(
        db_session,
        run_id=run_id,
        request=request,
        idempotency_key=f"run:{run_id}:kind:expand_plan_item:plan_item:implement-api",
    )

    assert same_key.id == first.id
    first_id = first.id
    first_target_id = first.target_id

    with pytest.raises(HTTPException) as exc_info:
        await service.execute_expand_plan_item_action(
            db_session,
            run_id=run_id,
            request=request,
            idempotency_key="run:phase12:kind:expand_plan_item:implement-api:retry",
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == f"Expand plan item idempotency key must be 'run:{run_id}:kind:expand_plan_item:plan_item:implement-api'"

    await db_session.flush()
    db_session.expire_all()

    persisted_run = await db_session.get(type(run), run_id)
    task = await db_session.get(Task, first_target_id)
    assert persisted_run is not None
    assert task is not None
    assert task.metadata_["orchestration"]["expand_action_id"] == str(first_id)
    expanded_items = persisted_run.plan_state["expanded_items"]
    assert len(expanded_items) == 1
    assert expanded_items[0]["plan_item_id"] == "implement-api"
    assert expanded_items[0]["work_function"] == "implementation"
    assert expanded_items[0]["expand_action_id"] == str(first_id)
    assert expanded_items[0]["task_id"] == str(task.id)
    assert expanded_items[0]["gate_id"] == task.metadata_["orchestration"]["plan_item_gate_id"]

    expand_actions = await _run_rows(db_session, OrchestrationAction, run_id, action_type="expand_plan_item")
    delegation_actions = await _run_rows(db_session, OrchestrationAction, run_id, action_type="create_delegation_task")
    gates = await _run_rows(db_session, OrchestrationGate, run_id, success_criterion_key="plan_item:implement-api")
    tasks = [await db_session.get(Task, action.target_id) for action in delegation_actions]
    events = await _event_rows(db_session, project_id, "orchestration.plan_item_expanded")

    assert len(expand_actions) == 1
    assert expand_actions[0].idempotency_key == f"run:{run_id}:kind:expand_plan_item:plan_item:implement-api"
    assert len(delegation_actions) == 1
    assert len(gates) == 1
    assert len([task for task in tasks if task is not None]) == 1
    assert len(events) == 1
    assert events[0].payload["action_id"] == str(first_id)


def test_plan_items_from_artifact_rejects_malformed_plan_item_shapes():
    artifact = Artifact(
        project_id=uuid.uuid4(),
        name="accepted-plan",
        artifact_type="plan",
        metadata_={"plan_items": [{**_raw_plan_item(), "scope": {"bad": "shape"}}]},
    )

    with pytest.raises(HTTPException) as exc_info:
        OrchestrationService()._plan_items_from_artifact(artifact)

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail.startswith("Invalid plan item at index 0 field scope:")


def test_plan_items_from_artifact_rejects_malformed_list_fields():
    artifact = Artifact(
        project_id=uuid.uuid4(),
        name="accepted-plan",
        artifact_type="plan",
        metadata_={"plan_items": [{**_raw_plan_item(), "inputs": "ROADMAP.md plan-to-work expansion"}]},
    )

    with pytest.raises(HTTPException) as exc_info:
        OrchestrationService()._plan_items_from_artifact(artifact)

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail.startswith("Invalid plan item at index 0 field inputs:")


@pytest.mark.asyncio
async def test_expand_accepted_plan_expands_three_items_and_replays_without_duplicates(db_session, test_project):
    planner = _agent("planner", role="planner", capabilities=["planning"])
    developer = _agent("developer", role="developer", capabilities=["implementation"])
    reviewer = _agent("reviewer", role="reviewer", capabilities=["review"])
    validator = _agent("validator", role="validator", capabilities=["validation"])
    db_session.add_all([planner, developer, reviewer, validator])
    await db_session.flush()
    plan_items = [
        _raw_plan_item(developer.id, item_id="implement-api", work_function="implementation"),
        _raw_plan_item(reviewer.id, item_id="review-api", work_function="review"),
        _raw_plan_item(validator.id, item_id="validate-api", work_function="validation"),
    ]
    service, _, run, _ = await _make_accepted_plan(db_session, test_project.id, planner, plan_items)

    first = await service.expand_accepted_plan(db_session, run.id)
    second = await service.expand_accepted_plan(db_session, run.id)

    assert [action.id for action in second] == [action.id for action in first]
    assert [action.action_type for action in first] == ["expand_plan_item", "expand_plan_item", "expand_plan_item"]
    assert all(action.status == "completed" for action in first)

    expand_actions = await _run_rows(db_session, OrchestrationAction, run.id, action_type="expand_plan_item")
    delegation_actions = await _run_rows(
        db_session,
        OrchestrationAction,
        run.id,
        action_type="create_delegation_task",
    )
    assert len(expand_actions) == 3
    assert len(delegation_actions) == 3

    tasks = list((await db_session.execute(select(Task).where(Task.project_id == test_project.id))).scalars().all())
    work_tasks = [
        task
        for task in tasks
        if isinstance(task.metadata_, dict)
        and isinstance(task.metadata_.get("orchestration"), dict)
        and task.metadata_["orchestration"].get("plan_item_id") in {"implement-api", "review-api", "validate-api"}
    ]
    assert len(work_tasks) == 3
    assert {task.assigned_to for task in work_tasks} == {developer.id, reviewer.id, validator.id}

    gate_result = await db_session.execute(
        select(OrchestrationGate).where(
            OrchestrationGate.run_id == run.id,
            OrchestrationGate.gate_type == "work_completed",
        )
    )
    gates = list(gate_result.scalars().all())
    assert len(gates) == 3
    assert {gate.success_criterion_key for gate in gates} == {
        "plan_item:implement-api",
        "plan_item:review-api",
        "plan_item:validate-api",
    }

    expanded_state = run.plan_state["expanded_items"]
    assert [item["plan_item_id"] for item in expanded_state] == ["implement-api", "review-api", "validate-api"]
    events = await _event_rows(db_session, test_project.id, "orchestration.plan_item_expanded")
    assert len(events) == 3


@pytest.mark.asyncio
async def test_expand_plan_item_without_agent_id_uses_strong_roster_fit(db_session, test_project):
    planner = _agent("planner", role="planner", capabilities=["planning"])
    developer = _agent("developer", role="developer", capabilities=["implementation"])
    db_session.add_all([planner, developer])
    await db_session.flush()
    service, _, run, _ = await _make_accepted_plan(
        db_session,
        test_project.id,
        planner,
        [_raw_plan_item(agent_id=None, item_id="implement-api", work_function="implementation")],
    )

    actions = await service.expand_accepted_plan(db_session, run.id)

    assert len(actions) == 1
    task = await db_session.get(Task, actions[0].target_id)
    assert task is not None
    assert task.assigned_to == developer.id


@pytest.mark.asyncio
async def test_expand_accepted_plan_rolls_back_when_roster_fit_is_weak(db_session, test_project):
    planner = _agent("planner", role="planner", capabilities=["planning"])
    developer = _agent("developer", role="developer", capabilities=["implementation"])
    db_session.add_all([planner, developer])
    await db_session.flush()
    service, _, run, _ = await _make_accepted_plan(
        db_session,
        test_project.id,
        planner,
        [
            _raw_plan_item(
                agent_id=developer.id,
                item_id="implement-api",
                work_function="implementation",
            ),
            {
                **_raw_plan_item(
                    agent_id=None,
                    item_id="missing-validation",
                    work_function="validation",
                ),
                "required_capabilities": [" validation "],
            },
        ],
    )
    run_id = run.id
    project_id = test_project.id

    with pytest.raises(HTTPException) as exc_info:
        await service.expand_accepted_plan(db_session, run_id)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "No strong roster fit for plan item 'missing-validation'"
    assert await _run_rows(db_session, OrchestrationAction, run_id, action_type="expand_plan_item") == []
    assert await _run_rows(db_session, OrchestrationAction, run_id, action_type="create_delegation_task") == []
    assert await _run_rows(db_session, OrchestrationGate, run_id, gate_type="work_completed") == []

    task_result = await db_session.execute(select(Task).where(Task.project_id == project_id))
    tasks = list(task_result.scalars().all())
    delegated_work_tasks = [
        task
        for task in tasks
        if isinstance(task.metadata_, dict)
        and isinstance(task.metadata_.get("orchestration"), dict)
        and task.metadata_["orchestration"].get("run_id") == str(run_id)
        and task.metadata_["orchestration"].get("plan_item_id") in {"implement-api", "missing-validation"}
    ]
    assert delegated_work_tasks == []

    events = await _event_rows(db_session, project_id, "orchestration.plan_item_expanded")
    assert [event for event in events if event.payload.get("run_id") == str(run_id)] == []
    db_session.expire_all()
    persisted_run = await db_session.get(type(run), run_id)
    assert persisted_run is not None
    assert "expanded_items" not in persisted_run.plan_state


@pytest.mark.asyncio
async def test_expand_accepted_plan_rolls_back_when_post_reserve_delegation_fails(
    db_session,
    test_project,
    monkeypatch,
):
    planner = _agent("planner", role="planner", capabilities=["planning"])
    developer = _agent("developer", role="developer", capabilities=["implementation"])
    reviewer = _agent("reviewer", role="reviewer", capabilities=["review"])
    db_session.add_all([planner, developer, reviewer])
    await db_session.flush()
    service, _, run, _ = await _make_accepted_plan(
        db_session,
        test_project.id,
        planner,
        [
            _raw_plan_item(developer.id, item_id="implement-api", work_function="implementation"),
            _raw_plan_item(reviewer.id, item_id="review-api", work_function="review"),
        ],
    )
    run_id = run.id
    project_id = test_project.id
    real_validate = service._validate_delegation_targets
    review_validate_calls = 0

    async def fail_review_post_reserve(db, run_id, request):
        nonlocal review_validate_calls
        if request.get("work_function") == "review":
            review_validate_calls += 1
            if review_validate_calls == 3:
                raise HTTPException(status_code=404, detail="Agent not found")
        return await real_validate(db, run_id, request)

    monkeypatch.setattr(service, "_validate_delegation_targets", fail_review_post_reserve)

    with pytest.raises(HTTPException) as exc_info:
        await service.expand_accepted_plan(db_session, run_id)

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Agent not found"
    assert review_validate_calls == 3
    assert await _run_rows(db_session, OrchestrationAction, run_id, action_type="expand_plan_item") == []
    assert await _run_rows(db_session, OrchestrationAction, run_id, action_type="create_delegation_task") == []
    assert await _run_rows(db_session, OrchestrationGate, run_id, gate_type="work_completed") == []

    task_result = await db_session.execute(select(Task).where(Task.project_id == project_id))
    tasks = list(task_result.scalars().all())
    delegated_work_tasks = [
        task
        for task in tasks
        if isinstance(task.metadata_, dict)
        and isinstance(task.metadata_.get("orchestration"), dict)
        and task.metadata_["orchestration"].get("run_id") == str(run_id)
        and task.metadata_["orchestration"].get("plan_item_id") in {"implement-api", "review-api"}
    ]
    assert delegated_work_tasks == []

    events = await _event_rows(db_session, project_id, "orchestration.plan_item_expanded")
    assert [event for event in events if event.payload.get("run_id") == str(run_id)] == []
    delegation_events = await _event_rows(db_session, project_id, "orchestration.delegation_task_created")
    assert [event for event in delegation_events if event.payload.get("run_id") == str(run_id)] == []
    db_session.expire_all()
    persisted_run = await db_session.get(type(run), run_id)
    assert persisted_run is not None
    assert "expanded_items" not in persisted_run.plan_state


@pytest.mark.asyncio
async def test_expand_accepted_plan_rejects_unaccepted_plan(db_session, test_project):
    planner = _agent("planner", role="planner", capabilities=["planning"])
    db_session.add(planner)
    await db_session.flush()
    service, _, run = await _make_run(db_session, test_project.id)
    await service.execute_request_plan_action(
        db_session,
        run_id=run.id,
        request={
            "action_type": "request_plan",
            "agent_id": str(planner.id),
            "work_function": "planning",
            "scope": "Create a plan with independently verifiable items.",
        },
        idempotency_key="run:phase12:kind:request_plan:unaccepted",
    )

    with pytest.raises(HTTPException) as exc_info:
        await service.expand_accepted_plan(db_session, run.id)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Plan must be accepted before expansion"
