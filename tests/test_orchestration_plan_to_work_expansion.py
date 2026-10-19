import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from huddleroom.models.agent import Agent
from huddleroom.models.artifact import Artifact
from huddleroom.models.event_log import EventLog
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGate, OrchestrationGoal
from huddleroom.models.orchestration_process import OrchestrationProcessRun
from huddleroom.models.task import Task
from huddleroom.schemas.orchestration import OrchestrationGoalCreate, OrchestrationPlanItem
from huddleroom.services.orchestration_plan_ingestion import plan_output_hash
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


async def _seed_accepted_g2_ownership(db_session, goal, planner, plan_items):
    """Give the candidate the accepted G2 facts its executable rows require."""
    agents = list((await db_session.scalars(select(Agent).where(Agent.is_active.is_(True)))).all())
    by_id = {str(agent.id): agent for agent in agents}
    roles = {"planning": str(planner.id)}
    eligibility = {str(planner.id): ["planning"]}
    coverage = [{
        "responsibility_id": "planning", "work_function": "planning", "agent_ref": str(planner.id),
        "status": "covered", "evidence_ref": "agent_review:accepted",
    }]
    for item in plan_items:
        function = item.get("work_function", "").strip()
        if not function or function in roles:
            continue
        agent_id = str(item.get("agent_id", "")).strip()
        agent = by_id.get(agent_id) or next(
            (candidate for candidate in agents if function in (candidate.capabilities or [])), None
        )
        if agent is None:
            continue
        roles[function] = str(agent.id)
        eligibility.setdefault(str(agent.id), []).append(function)
        coverage.append({
            "responsibility_id": function, "work_function": function, "agent_ref": str(agent.id),
            "status": "covered", "evidence_ref": "agent_review:accepted",
        })
    team = {
        "role_to_agent": roles,
        "approval": "approve",
        "approval_decision_id": str(uuid.uuid4()),
        "gates": {"team_structure_reviewed": True},
        "accepted_responsibility_gap_ids": [],
        "responsibility_coverage": coverage,
    }
    goal.orchestrator_context = {**(goal.orchestrator_context or {}), "team": team}
    for process_type, outputs in (
        ("agent_definition_review", {"eligibility": eligibility}),
        ("team_hierarchy", team),
    ):
        process = await db_session.scalar(select(OrchestrationProcessRun).where(
            OrchestrationProcessRun.goal_id == goal.id,
            OrchestrationProcessRun.process_type == process_type,
            OrchestrationProcessRun.superseded_by_id.is_(None),
        ))
        if process is None:
            process = OrchestrationProcessRun(
                goal_id=goal.id, process_type=process_type, status="completed", trigger_reason="test",
            )
            db_session.add(process)
        process.status = "completed"
        process.outputs = {**(process.outputs or {}), **outputs}
        process.skipped_by = None
        process.superseded_by_id = None
    await db_session.flush()


async def _candidate_artifact(db_session, project_id, run, planner, request_action, name, plan_items):
    normalized = OrchestrationService()._plan_items_from_raw(plan_items)
    envelope = {
        "schema_version": 1,
        "plan_kind": "implementation",
        "summary": "Implement the accepted plan.",
        "plan_items": [item.model_dump(mode="json") for item in normalized],
    }
    output_hash = plan_output_hash(envelope)
    artifact = Artifact(
        project_id=project_id,
        name=name,
        artifact_type="plan",
        status="draft",
        linked_task_id=request_action.target_id,
        created_by_agent=planner.id,
        content_hash=output_hash,
        metadata_={
            "kind": "implementation_plan",
            "source": "planning_output_ingestion",
            **envelope,
            "output_hash": output_hash,
            "winning_session_id": str(uuid.uuid4()),
            "planning_task_id": str(request_action.target_id),
            "planning_version": run.plan_state["planning_version"],
            "plan_author_agent_id": str(planner.id),
        },
    )
    db_session.add(artifact)
    await db_session.flush()
    run.plan_state = {
        **run.plan_state,
        "plan_artifact_id": str(artifact.id),
        "plan_output_hash": output_hash,
        "plan_provenance": {
            "artifact_id": str(artifact.id),
            "planning_task_id": str(request_action.target_id),
            "planning_version": run.plan_state["planning_version"],
            "winning_session_id": artifact.metadata_["winning_session_id"],
            "plan_author_agent_id": str(planner.id),
            "canonical_content_hash": output_hash,
        },
    }
    await db_session.flush()
    return artifact


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
    await _seed_accepted_g2_ownership(db_session, goal, planner, plan_items)
    artifact = await _candidate_artifact(
        db_session, project_id, run, planner, request_action, "expansion-plan", plan_items
    )
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
            "accepted_plan_fingerprint": run.plan_state["accepted_plan_snapshot"]["fingerprint"],
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
    assert expand_actions[0].idempotency_key == (
        f"run:{run_id}:kind:expand_plan_item:plan_item:implement-api:version:"
        f"{run.plan_state['accepted_plan_snapshot']['fingerprint']}"
    )
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
async def test_accept_plan_rejects_item_without_an_accepted_roster_owner(db_session, test_project):
    planner = _agent("planner", role="planner", capabilities=["planning"])
    developer = _agent("developer", role="developer", capabilities=["implementation"])
    db_session.add_all([planner, developer])
    await db_session.flush()
    service, goal, run = await _make_run(db_session, test_project.id)
    request_action = await service.execute_request_plan_action(
        db_session,
        run.id,
        {"action_type": "request_plan", "agent_id": str(planner.id), "work_function": "planning", "scope": "Plan."},
        f"run:{run.id}:kind:request_plan:missing-validation",
    )
    items = [
        _raw_plan_item(agent_id=developer.id, item_id="implement-api", work_function="implementation"),
        {**_raw_plan_item(agent_id=None, item_id="missing-validation", work_function="validation"),
         "required_capabilities": [" validation "]},
    ]
    await _seed_accepted_g2_ownership(db_session, goal, planner, items)
    artifact = await _candidate_artifact(db_session, test_project.id, run, planner, request_action, "weak-roster", items)
    run_id = run.id
    project_id = test_project.id

    with pytest.raises(HTTPException) as exc_info:
        await service.execute_accept_plan_action(
            db_session, run_id, {"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
            f"run:{run_id}:kind:accept_plan:missing-validation",
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Plan ownership is blocked: no accepted active agent for work_function 'validation'"
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


async def _adoption_setup(db_session, project_id):
    planner = _agent("planner", role="planner", capabilities=["planning"])
    developer = _agent("developer", role="developer", capabilities=["investigation"])
    db_session.add_all([planner, developer])
    await db_session.flush()
    service, goal, run = await _make_run(db_session, project_id)
    audit_action = await service.execute_create_delegation_task_action(
        db_session,
        run_id=run.id,
        request={
            "action_type": "create_delegation_task", "agent_id": str(developer.id),
            "work_function": "investigation", "scope": "Audit the API.", "deliverable": "Audit findings.",
        },
        idempotency_key=f"run:{run.id}:kind:create_delegation_task:adoption-audit",
    )
    task = await db_session.get(Task, audit_action.target_id)
    request_action = await service.execute_request_plan_action(
        db_session,
        run_id=run.id,
        request={
            "action_type": "request_plan",
            "agent_id": str(planner.id),
            "work_function": "planning",
            "scope": "Create a plan that adopts the running audit.",
        },
        idempotency_key="run:adoption:kind:request_plan",
    )
    return service, goal, run, planner, developer, request_action, task


def _adopting_item(task_id, item_id="audit-api", **overrides) -> dict:
    return {
        **_raw_plan_item(item_id=item_id, work_function="investigation"),
        "adopt_task_id": str(task_id),
        **overrides,
    }


async def _accept_with_items(db_session, project_id, service, run, planner, request_action, items):
    goal = await db_session.get(OrchestrationGoal, run.goal_id)
    assert goal is not None
    await _seed_accepted_g2_ownership(db_session, goal, planner, items)
    artifact = await _candidate_artifact(
        db_session, project_id, run, planner, request_action, "adoption-plan", items
    )
    return await service.execute_accept_plan_action(
        db_session,
        run_id=run.id,
        request={"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
        idempotency_key=f"run:{run.id}:kind:accept_plan:{artifact.id}",
    )


async def _task_count(db_session, project_id) -> int:
    return len((await db_session.execute(select(Task.id).where(Task.project_id == project_id))).scalars().all())


async def _assert_adoption_rejected(db_session, project_id, service, run, planner, request_action, items, reason):
    tasks_before = await _task_count(db_session, project_id)
    with pytest.raises(HTTPException) as exc_info:
        await _accept_with_items(db_session, project_id, service, run, planner, request_action, items)
    assert exc_info.value.status_code == 409
    assert exc_info.value.detail.startswith("adoption_invalid: item ")
    assert reason in exc_info.value.detail
    assert run.plan_state["status"] == "revision_required"
    assert "accepted_artifact_id" not in run.plan_state
    assert await _run_rows(db_session, OrchestrationGate, run.id, gate_type="work_completed") == []
    assert await _task_count(db_session, project_id) == tasks_before


@pytest.mark.asyncio
async def test_accept_plan_accepts_valid_adoption_without_creating_work(db_session, test_project):
    service, _, run, planner, _, request_action, task = await _adoption_setup(db_session, test_project.id)
    tasks_before = await _task_count(db_session, test_project.id)

    action = await _accept_with_items(
        db_session, test_project.id, service, run, planner, request_action,
        [_adopting_item(task.id)],
    )

    assert action.status == "completed"
    assert run.plan_state["status"] == "accepted"
    assert run.plan_state["accepted_plan_snapshot"]["items"][0]["adopt_task_id"] == str(task.id)
    assert await _task_count(db_session, test_project.id) == tasks_before


@pytest.mark.asyncio
async def test_accept_plan_allows_adopted_task_already_bound_to_the_same_item_gate(db_session, test_project):
    service, _, run, planner, _, request_action, task = await _adoption_setup(db_session, test_project.id)
    gate = await service._ensure_plan_item_gate(
        db_session, run.id, OrchestrationPlanItem.model_validate(_adopting_item(task.id))
    )
    task.metadata_ = {
        **task.metadata_,
        "orchestration": {**task.metadata_["orchestration"], "plan_item_gate_id": str(gate.id)},
    }
    await db_session.flush()

    action = await _accept_with_items(
        db_session, test_project.id, service, run, planner, request_action, [_adopting_item(task.id)]
    )

    assert action.status == "completed"


@pytest.mark.asyncio
async def test_accept_plan_rejects_task_adopted_by_two_items(db_session, test_project):
    service, _, run, planner, _, request_action, task = await _adoption_setup(db_session, test_project.id)
    await _assert_adoption_rejected(
        db_session, test_project.id, service, run, planner, request_action,
        [_adopting_item(task.id, "audit-a"), _adopting_item(task.id, "audit-b")],
        "adopted by more than one item",
    )


@pytest.mark.asyncio
async def test_accept_plan_rejects_adopting_work_of_another_run(db_session, test_project):
    service, _, run, planner, _, request_action, task = await _adoption_setup(db_session, test_project.id)
    task.metadata_ = {
        **task.metadata_,
        "orchestration": {**task.metadata_["orchestration"], "run_id": str(uuid.uuid4())},
    }
    await db_session.flush()
    await _assert_adoption_rejected(
        db_session, test_project.id, service, run, planner, request_action,
        [_adopting_item(task.id)], "is not work of this run",
    )


@pytest.mark.asyncio
async def test_accept_plan_rejects_adopting_missing_task(db_session, test_project):
    service, _, run, planner, _, request_action, _ = await _adoption_setup(db_session, test_project.id)
    await _assert_adoption_rejected(
        db_session, test_project.id, service, run, planner, request_action,
        [_adopting_item(uuid.uuid4())], "is not work of this run",
    )


@pytest.mark.asyncio
async def test_accept_plan_rejects_adopting_cancelled_task(db_session, test_project):
    service, _, run, planner, _, request_action, task = await _adoption_setup(db_session, test_project.id)
    task.status = "cancelled"
    await db_session.flush()
    await _assert_adoption_rejected(
        db_session, test_project.id, service, run, planner, request_action,
        [_adopting_item(task.id)], "is cancelled",
    )


@pytest.mark.asyncio
async def test_accept_plan_rejects_adopting_task_without_delegation_contract(db_session, test_project):
    service, _, run, planner, _, request_action, task = await _adoption_setup(db_session, test_project.id)
    task.metadata_ = {"orchestration": task.metadata_["orchestration"]}
    await db_session.flush()
    await _assert_adoption_rejected(
        db_session, test_project.id, service, run, planner, request_action,
        [_adopting_item(task.id)], "was not created by a delegation",
    )


@pytest.mark.asyncio
async def test_accept_plan_rejects_adoption_with_mismatched_work_function(db_session, test_project):
    service, _, run, planner, _, request_action, task = await _adoption_setup(db_session, test_project.id)
    await _assert_adoption_rejected(
        db_session, test_project.id, service, run, planner, request_action,
        [_adopting_item(task.id, work_function="implementation")],
        "work_function 'investigation' does not match 'implementation'",
    )


@pytest.mark.asyncio
async def test_accept_plan_rejects_adopting_non_adoptable_work_function(db_session, test_project):
    service, _, run, planner, _, request_action, task = await _adoption_setup(db_session, test_project.id)
    task.metadata_ = {
        **task.metadata_,
        "orchestration": {**task.metadata_["orchestration"], "work_function": "verification"},
    }
    await db_session.flush()
    await _assert_adoption_rejected(
        db_session, test_project.id, service, run, planner, request_action,
        [_adopting_item(task.id, work_function="verification")],
        "non-adoptable work_function 'verification'",
    )


@pytest.mark.asyncio
async def test_accept_plan_rejects_adoption_assigned_to_another_agent(db_session, test_project):
    service, _, run, planner, _, request_action, task = await _adoption_setup(db_session, test_project.id)
    await _assert_adoption_rejected(
        db_session, test_project.id, service, run, planner, request_action,
        [_adopting_item(task.id, agent_id=str(planner.id))], "assigned to another agent",
    )


@pytest.mark.asyncio
async def test_accept_plan_rejects_adopting_task_bound_to_another_gate(db_session, test_project):
    service, _, run, planner, _, request_action, task = await _adoption_setup(db_session, test_project.id)
    other_gate = OrchestrationGate(
        run_id=run.id, success_criterion_key="plan_item:other", gate_type="work_completed", required_evidence={}
    )
    db_session.add(other_gate)
    await db_session.flush()
    task.metadata_ = {
        **task.metadata_,
        "orchestration": {**task.metadata_["orchestration"], "plan_item_gate_id": str(other_gate.id)},
    }
    await db_session.flush()
    with pytest.raises(HTTPException) as exc_info:
        await _accept_with_items(
            db_session, test_project.id, service, run, planner, request_action, [_adopting_item(task.id)]
        )
    assert exc_info.value.detail == f"adoption_invalid: item audit-api: task {task.id} is already bound to another gate"
    assert run.plan_state["status"] == "revision_required"
    assert [g.id for g in await _run_rows(db_session, OrchestrationGate, run.id, gate_type="work_completed")] == [
        other_gate.id
    ]


@pytest.mark.asyncio
async def test_accept_plan_rejects_adopting_item_with_dependencies(db_session, test_project):
    service, _, run, planner, developer, request_action, task = await _adoption_setup(db_session, test_project.id)
    await _assert_adoption_rejected(
        db_session, test_project.id, service, run, planner, request_action,
        [_raw_plan_item(developer.id, item_id="first"), _adopting_item(task.id, depends_on=["first"])],
        "an adopting item cannot have depends_on",
    )


@pytest.mark.asyncio
async def test_validate_plan_adoptions_rejects_goal_and_shared_state_roadmap_items(db_session, test_project):
    from huddleroom.schemas.orchestration import OrchestrationRoadmapPlanItem

    service, _, run, _, _, _, task = await _adoption_setup(db_session, test_project.id)
    goal_item = OrchestrationRoadmapPlanItem.model_validate({
        "item_key": "child", "unit_type": "goal", "title": "Child", "objective": "o",
        "success_criteria": [{"key": "k", "description": "d"}], "allocation": {"max_tokens": 10},
        "adopt_task_id": str(task.id),
    })
    shared_item = OrchestrationRoadmapPlanItem.model_validate({
        "item_key": "shared", "unit_type": "task", "title": "Shared", "work_function": "investigation",
        "scope": "s", "deliverable": "d", "adopt_task_id": str(task.id),
    })
    unshared_item = shared_item.model_copy(update={"mutates_shared_state": False})

    for item in (goal_item, shared_item):
        with pytest.raises(HTTPException) as exc_info:
            await service._validate_plan_adoptions(db_session, run, [item])
        assert exc_info.value.detail == (
            f"adoption_invalid: item {item.item_key}: "
            "only Roadmap task items with mutates_shared_state false can adopt work"
        )
    await service._validate_plan_adoptions(db_session, run, [unshared_item])


async def _adopted_audit_plan(db_session, project_id, *, audits=1, synthesis=None, ad_hoc=False):
    """Delegated pre-plan audits, then a request_plan + accepted plan adopting each one.

    `synthesis` (optional item overrides) adds a join item run by the auditor that depends on every audit."""
    from tests.conftest import bind_plan_author_team

    planner = _agent("planner", role="planner", capabilities=["planning"])
    auditor = _agent("auditor", role="developer", capabilities=["investigation"])
    db_session.add_all([planner, auditor])
    await db_session.flush()
    service, goal, run = await _make_run(db_session, project_id)
    tasks = []
    for n in range(audits):
        action = await service.execute_create_delegation_task_action(
            db_session,
            run_id=run.id,
            ad_hoc=ad_hoc,
            request={
                "action_type": "create_delegation_task",
                "agent_id": str(auditor.id),
                "work_function": "investigation",
                "scope": f"Audit area {n}.",
                "deliverable": f"Audit findings {n}.",
            },
            idempotency_key=f"run:{run.id}:kind:create_delegation_task:audit-{n}",
        )
        tasks.append(await db_session.get(Task, action.target_id))
    await bind_plan_author_team(db_session, goal, planner)
    request_action = await service.execute_request_plan_action(
        db_session,
        run_id=run.id,
        request={
            "action_type": "request_plan",
            "agent_id": str(planner.id),
            "work_function": "planning",
            "scope": "Plan around the running audits.",
        },
        idempotency_key="run:adopted-audits:kind:request_plan",
    )
    items = [_adopting_item(task.id, item_id=f"audit-{n}", priority=70 + n) for n, task in enumerate(tasks)]
    if synthesis is not None:
        items.append({
            **_raw_plan_item(auditor.id, item_id="synthesis", work_function="investigation"),
            "depends_on": [f"audit-{n}" for n in range(audits)],
            **synthesis,
        })
    action = await _accept_with_items(db_session, project_id, service, run, planner, request_action, items)
    assert action.status == "completed"
    return service, goal, run, tasks


async def _expand(service, db_session, run, item_id, work_function="investigation"):
    return await service.execute_expand_plan_item_action(
        db_session,
        run_id=run.id,
        request={"action_type": "expand_plan_item", "plan_item_id": item_id, "work_function": work_function},
        idempotency_key=f"run:{run.id}:kind:expand_plan_item:plan_item:{item_id}",
    )


@pytest.mark.asyncio
async def test_expanding_adopted_item_binds_active_task_without_new_work(db_session, test_project):
    from huddleroom.models.session import Session

    service, goal, run, (audit,) = await _adopted_audit_plan(db_session, test_project.id)
    audit.status = "in_progress"
    session = Session(
        task_id=audit.id, agent_id=audit.assigned_to, project_id=test_project.id, adapter_type="api", status="running"
    )
    db_session.add(session)
    await db_session.flush()
    tasks_before = await _task_count(db_session, test_project.id)

    action = await _expand(service, db_session, run, "audit-0")

    assert action.status == "completed" and action.target_id == audit.id
    assert await _task_count(db_session, test_project.id) == tasks_before
    assert [s.id for s in await db_session.scalars(select(Session).where(Session.task_id == audit.id))] == [session.id]
    gate = (await _run_rows(db_session, OrchestrationGate, run.id, success_criterion_key="plan_item:audit-0"))[0]
    assert audit.metadata_["orchestration"]["plan_item_gate_id"] == str(gate.id)
    assert audit.metadata_["orchestration"]["plan_item_id"] == "audit-0"
    assert audit.priority == 70 and audit.status == "in_progress"
    assert await _run_rows(db_session, OrchestrationAction, run.id, action_type="create_delegation_task") == [
        await db_session.get(OrchestrationAction, audit.id)
    ]
    assert run.plan_state["expanded_items"] == [{
        "plan_item_id": "audit-0",
        "work_function": "investigation",
        "expand_action_id": str(action.id),
        "delegation_action_id": str(audit.id),
        "task_id": str(audit.id),
        "gate_id": str(gate.id),
        "adopted": True,
        "accepted_plan_fingerprint": run.plan_state["accepted_plan_snapshot"]["fingerprint"],
    }]
    (event,) = await _event_rows(db_session, test_project.id, "orchestration.plan_item_expanded")
    assert event.payload["adopted"] is True and event.payload["task_id"] == str(audit.id)
    (marker,) = await _run_rows(db_session, OrchestrationAction, run.id, action_type="adopt_task")
    assert marker.idempotency_key == f"run:{run.id}:kind:adopt_task:task:{audit.id}"
    assert marker.status == "completed" and marker.target_id == audit.id
    assert marker.request == {
        "plan_item_id": "audit-0",
        "adopt_task_id": str(audit.id),
        "accepted_plan_fingerprint": run.plan_state["accepted_plan_snapshot"]["fingerprint"],
    }


@pytest.mark.asyncio
async def test_expanding_adopted_item_replays_terminal_task_evidence_and_consumes_report(db_session, test_project):
    from huddleroom.models.orchestration import OrchestrationEvidence
    from huddleroom.models.session import Session
    from huddleroom.services.event_bus import emit_event_once

    service, goal, run, (audit,) = await _adopted_audit_plan(db_session, test_project.id)
    audit.status = "done"
    session = Session(
        task_id=audit.id, agent_id=audit.assigned_to, project_id=test_project.id, adapter_type="api",
        status="completed", output="Audit finished.",
    )
    db_session.add(session)
    await db_session.flush()
    await emit_event_once(
        db_session, test_project.id, "task.status_changed",
        {"task_id": str(audit.id), "status": "done", "previous_status": "in_progress"}, dedup_key=f"done:{audit.id}",
    )
    await emit_event_once(
        db_session, test_project.id, "session.completed",
        {"session_id": str(session.id), "task_id": str(audit.id)}, dedup_key=f"completed:{session.id}",
    )
    # Before adoption the gate-less audit's completion was skipped by evidence ingestion.
    tasks_before = await _task_count(db_session, test_project.id)

    await _expand(service, db_session, run, "audit-0")

    gate = (await _run_rows(db_session, OrchestrationGate, run.id, success_criterion_key="plan_item:audit-0"))[0]
    evidence = await _run_rows(db_session, OrchestrationEvidence, run.id, gate_id=gate.id)
    assert sorted((e.source_type, e.source_id) for e in evidence) == sorted(
        [("task", audit.id), ("session", session.id)]
    )
    marker = await service._existing_action_for_key(
        db_session, run.id, f"run:{run.id}:kind:report_consumed:task:{audit.id}"
    )
    assert marker is not None
    # Adoption creates no work; the prose report gets exactly one output-only correction task.
    assert await _task_count(db_session, test_project.id) == tasks_before + 1
    assert marker.request["result_correction"]["status"] == "correcting"


@pytest.mark.asyncio
async def test_expanding_adopted_item_twice_binds_once(db_session, test_project):
    service, goal, run, (audit,) = await _adopted_audit_plan(db_session, test_project.id)
    tasks_before = await _task_count(db_session, test_project.id)

    first = await _expand(service, db_session, run, "audit-0")
    second = await _expand(service, db_session, run, "audit-0")
    await service._release_ready_work(db_session, goal, run)

    assert second.id == first.id
    assert len(await _run_rows(db_session, OrchestrationAction, run.id, action_type="adopt_task")) == 1
    assert len(await _run_rows(db_session, OrchestrationAction, run.id, action_type="expand_plan_item")) == 1
    assert [e["task_id"] for e in run.plan_state["expanded_items"]] == [str(audit.id)]
    assert len(await _run_rows(db_session, OrchestrationGate, run.id, gate_type="work_completed")) == 1
    assert await _task_count(db_session, test_project.id) == tasks_before


@pytest.mark.asyncio
async def test_adopted_item_replays_from_a_fresh_session_without_new_effects(db_session, test_engine, test_project):
    service, goal, run, (audit,) = await _adopted_audit_plan(db_session, test_project.id)
    first = await _expand(service, db_session, run, "audit-0")
    await db_session.commit()

    sessions = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with sessions.begin() as replay_db:
        persisted_goal = await replay_db.get(type(goal), goal.id)
        persisted_run = await replay_db.get(type(run), run.id)
        persisted_audit = await replay_db.get(Task, audit.id)
        assert persisted_goal is not None and persisted_run is not None and persisted_audit is not None
        replay = await _expand(OrchestrationService(), replay_db, persisted_run, "audit-0")
        assert replay.id == first.id
        assert persisted_audit.metadata_["orchestration_plan_item"]["agent_id"] is None
        assert persisted_audit.assigned_to == audit.assigned_to
        assert len(await _run_rows(replay_db, OrchestrationAction, persisted_run.id, action_type="adopt_task")) == 1
        assert len(await _run_rows(replay_db, OrchestrationAction, persisted_run.id, action_type="expand_plan_item")) == 1
        assert len(await _run_rows(replay_db, OrchestrationGate, persisted_run.id, gate_type="work_completed")) == 1


@pytest.mark.asyncio
async def test_frozen_adoption_owner_cannot_adopt_work_assigned_to_another_agent(db_session, test_project):
    service, _, run, planner, developer, request_action, task = await _adoption_setup(db_session, test_project.id)
    task.assigned_to = planner.id
    await db_session.flush()
    await _accept_with_items(
        db_session, test_project.id, service, run, planner, request_action,
        [_adopting_item(task.id, agent_id=str(planner.id))],
    )
    assert run.plan_state["accepted_plan_snapshot"]["items"][0]["agent_id"] == str(planner.id)
    task.assigned_to = developer.id
    await db_session.flush()

    with pytest.raises(HTTPException) as exc_info:
        await _expand(service, db_session, run, "audit-api")

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail.startswith("adoption_invalid:")
    assert await _run_rows(db_session, OrchestrationAction, run.id, action_type="adopt_task") == []
    assert await _run_rows(db_session, OrchestrationAction, run.id, action_type="expand_plan_item") == []
    assert await _run_rows(db_session, OrchestrationGate, run.id, gate_type="work_completed") == []
    assert task.assigned_to == developer.id


@pytest.mark.asyncio
async def test_adoption_marker_for_another_item_conflicts(db_session, test_project):
    service, goal, run, (audit,) = await _adopted_audit_plan(db_session, test_project.id)
    await service.reserve_action(
        db_session, run_id=run.id, idempotency_key=f"run:{run.id}:kind:adopt_task:task:{audit.id}",
        action_type="adopt_task", request={"plan_item_id": "older", "adopt_task_id": str(audit.id),
                                           "accepted_plan_fingerprint": "older"},
    )

    with pytest.raises(HTTPException) as exc_info:
        await _expand(service, db_session, run, "audit-0")

    assert exc_info.value.status_code == 409 and exc_info.value.detail.startswith("adoption_conflict:")
    assert "plan_item_gate_id" not in audit.metadata_["orchestration"]


@pytest.mark.asyncio
async def test_release_binds_adopted_items_without_spending_release_slots(db_session, test_project):
    service, goal, run, audits = await _adopted_audit_plan(db_session, test_project.id, audits=3)
    tasks_before = await _task_count(db_session, test_project.id)

    assert await service._release_ready_work(db_session, goal, run) == 3
    assert await service._release_ready_work(db_session, goal, run) == 0

    assert sorted(e["task_id"] for e in run.plan_state["expanded_items"]) == sorted(str(t.id) for t in audits)
    assert len(await _run_rows(db_session, OrchestrationAction, run.id, action_type="release_item")) == 3
    assert await _task_count(db_session, test_project.id) == tasks_before


@pytest.mark.asyncio
async def test_unadopted_pre_plan_audit_must_be_explained_and_then_continues_unbound(db_session, test_project):
    import json

    from huddleroom.models.session import Session
    from tests.conftest import bind_plan_author_team

    planner = _agent("planner", role="planner", capabilities=["planning"])
    auditor = _agent("auditor", role="developer", capabilities=["investigation"])
    db_session.add_all([planner, auditor])
    await db_session.flush()
    service, goal, run = await _make_run(db_session, test_project.id)
    audit_action = await service.execute_create_delegation_task_action(
        db_session, run_id=run.id,
        request={"action_type": "create_delegation_task", "agent_id": str(auditor.id),
                 "work_function": "investigation", "scope": "Audit the API.", "deliverable": "Findings."},
        idempotency_key=f"run:{run.id}:kind:create_delegation_task:disposition-audit",
    )
    audit = await db_session.get(Task, audit_action.target_id)
    audit.status = "in_progress"
    await bind_plan_author_team(db_session, goal, planner)
    await service.execute_request_plan_action(
        db_session, run_id=run.id,
        request={"action_type": "request_plan", "agent_id": str(planner.id), "work_function": "planning",
                 "scope": "Plan the work."},
        idempotency_key="run:disposition:kind:request_plan",
    )

    async def finish_planning(summary):
        task = await db_session.get(Task, uuid.UUID(run.plan_state["planning_task_id"]))
        assert task.metadata_["orchestration_plan"]["pre_plan_task_ids"] == [str(audit.id)]
        session = Session(
            task_id=task.id, agent_id=planner.id, project_id=test_project.id, adapter_type="api",
            status="completed", output=json.dumps({
                "schema_version": 1,
                "plan_kind": task.metadata_["orchestration_contract"]["report_schema"]["plan_kind"],
                "summary": summary,
                "plan_items": [{**_raw_plan_item(planner.id), "why_runnable": "No dependencies."}],
            }),
        )
        task.status = "done"
        db_session.add(session)
        await db_session.flush()
        await service._ingest_planning_output(db_session, run, task, session.id)
        return task

    first = await finish_planning("Implement the API.")
    assert run.plan_state["status"] == "revision_required"
    assert run.plan_state["revision_reason"] == (
        f"planning_output_rejected: pre-plan task {audit.id} is neither adopted (adopt_task_id) nor explained in summary"
    )
    assert "plan_artifact_id" not in run.plan_state

    await service.execute_request_plan_revision_action(
        db_session, run_id=run.id,
        request={"action_type": "request_plan_revision", "plan_task_id": str(first.id),
                 "revision_request": "Disposition the running audit."},
        idempotency_key="run:disposition:kind:request_plan_revision",
    )
    await finish_planning(f"Implement the API; audit {audit.id} keeps running as independent context.")
    artifact = await db_session.get(Artifact, uuid.UUID(run.plan_state["plan_artifact_id"]))
    assert artifact is not None
    await _seed_accepted_g2_ownership(db_session, goal, planner, artifact.metadata_["plan_items"])
    await service.execute_accept_plan_action(
        db_session, run_id=run.id,
        request={"action_type": "accept_plan", "plan_artifact_id": run.plan_state["plan_artifact_id"]},
        idempotency_key="run:disposition:kind:accept_plan",
    )
    assert run.plan_state["status"] == "accepted"

    assert await service._release_ready_work(db_session, goal, run) == 1
    await db_session.refresh(audit)
    assert audit.status == "in_progress"
    assert "plan_item_gate_id" not in audit.metadata_["orchestration"]
    assert "orchestration_plan_item" not in audit.metadata_
    assert [e["plan_item_id"] for e in run.plan_state["expanded_items"]] == ["implement-api"]
    assert await _run_rows(db_session, OrchestrationAction, run.id, action_type="adopt_task") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["in_progress", "done"])
async def test_compatible_second_plan_refreshes_bound_task_without_rewriting_work(db_session, test_project, status):
    """Compatible refresh preserves work and accepted historical evidence."""
    from copy import deepcopy
    from huddleroom.models.session import Session
    from huddleroom.models.orchestration import OrchestrationEvidence
    service, goal, run, (audit,) = await _adopted_audit_plan(db_session, test_project.id)
    await _expand(service, db_session, run, "audit-0")
    audit.status = status
    contract_before = deepcopy(audit.metadata_["orchestration_contract"])
    gates_before = [row.id for row in await _run_rows(db_session, OrchestrationGate, run.id)]
    evidence_before = [row.id for row in await _run_rows(db_session, OrchestrationEvidence, run.id)]
    session = Session(task_id=audit.id, agent_id=audit.assigned_to, project_id=test_project.id,
                      adapter_type="api", status="running" if status == "in_progress" else "completed")
    artifact = Artifact(project_id=test_project.id, name="reference", artifact_type="document", status="draft")
    db_session.add_all([session, artifact])
    await db_session.flush()
    task_count = await _task_count(db_session, test_project.id)
    request_action = (await _run_rows(db_session, OrchestrationAction, run.id, action_type="request_plan"))[0]
    planner = await db_session.get(Agent, uuid.UUID(request_action.request["agent_id"]))

    # A replan: put the run back in the revision state a request_plan_revision would leave it in.
    run.plan_state = {**run.plan_state, "status": "revision_requested", "planning_version": 2}
    await db_session.flush()
    # New plan metadata may refresh the stable binding.
    await _accept_with_items(
        db_session, test_project.id, service, run, planner, request_action,
        [_adopting_item(audit.id, item_id="audit-0", priority=99, artifact_ids=[str(artifact.id)])],
    )
    (marker,) = await _run_rows(db_session, OrchestrationAction, run.id, action_type="adopt_task")
    v1_request = dict(marker.request)
    # Replaying this accepted version must preserve its existing release.
    refreshed = await _expand(service, db_session, run, "audit-0")
    assert (await _expand(service, db_session, run, "audit-0")).id == refreshed.id
    assert audit.priority == 99
    assert audit.status == status and audit.metadata_["orchestration_contract"] == contract_before
    assert str(artifact.id) in audit.metadata_["orchestration_plan_item"]["artifact_ids"]
    assert await _task_count(db_session, test_project.id) == task_count
    assert [row.id for row in await _run_rows(db_session, OrchestrationGate, run.id)] == gates_before
    assert set(evidence_before).issubset({row.id for row in await _run_rows(db_session, OrchestrationEvidence, run.id)})
    assert list(await db_session.scalars(select(Session).where(Session.task_id == audit.id))) == [session]
    (marker,) = await _run_rows(db_session, OrchestrationAction, run.id, action_type="adopt_task")
    assert marker.request["plan_item_id"] == v1_request["plan_item_id"]

    run.plan_state = {**run.plan_state, "status": "revision_requested", "planning_version": 3}
    await db_session.flush()
    # Different item id: the task is already bound to audit-0's gate, so acceptance rejects it.
    with pytest.raises(HTTPException) as exc_info:
        await _accept_with_items(
            db_session, test_project.id, service, run, planner, request_action,
            [_adopting_item(audit.id, item_id="renamed", priority=10)],
        )
    assert exc_info.value.status_code == 409 and exc_info.value.detail.startswith("adoption_invalid:")


@pytest.mark.asyncio
async def test_release_blocks_plan_integrity_when_adopted_binding_fails(db_session, test_project):
    service, goal, run, (audit,) = await _adopted_audit_plan(db_session, test_project.id)
    await service.reserve_action(
        db_session, run_id=run.id, idempotency_key=f"run:{run.id}:kind:adopt_task:task:{audit.id}",
        action_type="adopt_task", request={"plan_item_id": "older", "adopt_task_id": str(audit.id),
                                           "accepted_plan_fingerprint": "older"},
    )

    for _ in range(2):  # the failed expansion replays; it never raises out of release
        assert await service._release_ready_work(db_session, goal, run) == 0

    assert await _run_rows(db_session, OrchestrationAction, run.id, action_type="release_item") == []
    (expand,) = await _run_rows(db_session, OrchestrationAction, run.id, action_type="expand_plan_item")
    assert expand.status == "failed" and expand.error.startswith("adoption_conflict:")
    assert not run.plan_state.get("expanded_items")
    (blocker,) = [b for b in run.active_blockers if b.get("kind") == "plan_criterion_integrity"]
    assert blocker["reason"].startswith("Adopted plan item 'audit-0' could not bind: adoption_conflict:")
    assert "plan_item_gate_id" not in audit.metadata_["orchestration"]


@pytest.mark.asyncio
async def test_accept_plan_rejects_item_artifact_outside_project(db_session, test_project):
    service, _, run, planner, _, request_action, task = await _adoption_setup(db_session, test_project.id)
    ghost = uuid.uuid4()
    tasks_before = await _task_count(db_session, test_project.id)

    with pytest.raises(HTTPException) as exc_info:
        await _accept_with_items(
            db_session, test_project.id, service, run, planner, request_action,
            [_adopting_item(task.id, artifact_ids=[str(ghost)])],
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == f"artifact_invalid: item audit-api: artifact {ghost} not found in project"
    assert run.plan_state["status"] == "revision_required"
    assert await _task_count(db_session, test_project.id) == tasks_before


@pytest.mark.asyncio
async def test_adopting_a_gated_ad_hoc_audit_hands_over_to_the_item_gate(db_session, test_project):
    from huddleroom.services.orchestration_dependencies import ADHOC_GATE_TYPE, task_gate_id

    service, goal, run, (audit,) = await _adopted_audit_plan(db_session, test_project.id, ad_hoc=True)
    adhoc_gate_id = audit.metadata_["orchestration"]["gate_id"]  # rule 7 let the plan be accepted
    assert (await db_session.get(OrchestrationGate, uuid.UUID(adhoc_gate_id))).gate_type == ADHOC_GATE_TYPE

    action = await _expand(service, db_session, run, "audit-0")

    assert action.status == "completed" and action.target_id == audit.id
    item_gate = (await _run_rows(db_session, OrchestrationGate, run.id, success_criterion_key="plan_item:audit-0"))[0]
    orchestration = audit.metadata_["orchestration"]
    assert "gate_id" not in orchestration and orchestration["adhoc_gate_id"] == adhoc_gate_id
    assert orchestration["plan_item_gate_id"] == str(item_gate.id)
    assert task_gate_id(audit, run.id) == item_gate.id
    assert run.plan_state["expanded_items"][0]["gate_id"] == str(item_gate.id)
    assert (await db_session.get(OrchestrationGate, uuid.UUID(adhoc_gate_id))) is not None


@pytest.mark.asyncio
async def test_second_plan_rejects_substantive_adoption_before_mutating_binding(db_session, test_project):
    from copy import deepcopy
    service, goal, run, (audit,) = await _adopted_audit_plan(db_session, test_project.id)
    await _expand(service, db_session, run, "audit-0")
    before = deepcopy(audit.metadata_)
    request = (await _run_rows(db_session, OrchestrationAction, run.id, action_type="request_plan"))[0]
    planner = await db_session.get(Agent, uuid.UUID(request.request["agent_id"]))
    run.plan_state = {**run.plan_state, "status": "revision_requested"}
    item = {**_adopting_item(audit.id, item_id="audit-0"), "scope": "Different substantive work"}
    with pytest.raises(HTTPException):
        await _accept_with_items(db_session, test_project.id, service, run, planner, request, [item])
    assert audit.metadata_ == before
