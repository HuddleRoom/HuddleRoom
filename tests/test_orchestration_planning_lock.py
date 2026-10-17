"""File-backed SQLite regressions for planner ingestion followed by reconciliation."""
# pylint: disable=too-many-locals,not-callable
import json

import pytest
from sqlalchemy import event, func, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from huddleroom.models.artifact import Artifact
from huddleroom.models.orchestration import OrchestrationDecision, OrchestrationGoal, OrchestrationRun
from huddleroom.models.project import Project
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.services.event_bus import emit_event_once
from huddleroom.services.orchestration_llm_decision_adapter import OrchestrationDecisionAdapter
from huddleroom.services.orchestration_plan_ingestion import ingest
from tests.orchestration_wake_helpers import NOOP_WAKE_WHEN
from tests.test_orchestration_plan_ingestion import _db_output, _env, _finish, _item, _planning_run
from tests.test_orchestration_runtime_e2e import _agent, _authorized_run


@pytest.mark.asyncio
async def test_planner_terminal_tick_releases_sqlite_writer_before_controller(test_engine, tmp_path, monkeypatch):
    """A successful Artifact insert must close its savepoint before the model boundary.

    Use committed rows and independent connections to the file-backed test database;
    a transaction-wrapped db_session would hide the production ownership semantics.
    """
    if test_engine.dialect.name != "sqlite":
        pytest.skip("SQLite single-writer regression")
    monkeypatch.setattr("huddleroom.workers.task_runner.register_session", lambda *_args, **_kwargs: None)
    import huddleroom.services.orchestration_service as service_module

    assert test_engine.url.database != ":memory:"
    engine = create_async_engine(test_engine.url, connect_args={"timeout": 0.1})
    factory = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(service_module, "AsyncSessionLocal", factory)
    # Make a regression fail promptly; production keeps its existing busy timeout.
    def short_busy_timeout(connection, _record):
        cursor = connection.cursor()
        cursor.execute("PRAGMA busy_timeout=100")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    event.listen(engine.sync_engine, "connect", short_busy_timeout)
    try:
        async with factory() as setup:
            project = Project(name="Planning lock regression", workspace_path=str(tmp_path), config={})
            setup.add(project)
            await setup.flush()
            project_id = project.id
            planner = _agent("lock-planner", ["planning"])
            setup.add(planner)
            await setup.flush()
            service, _, run = await _authorized_run(setup, project, owns_setup=True)
            action = await service.execute_request_plan_action(setup, run.id, {
                "action_type": "request_plan", "agent_id": str(planner.id),
                "work_function": "planning", "scope": "Plan the fixture audit.",
            }, f"run:{run.id}:kind:lock-plan")
            task = await setup.get(Task, action.target_id)
            session = Session(task_id=task.id, agent_id=planner.id, project_id=project_id,
                              adapter_type="api", status="completed")
            setup.add(session)
            await _finish(setup, task, session, json.dumps(_env(items=[_item(success_criterion_keys=["done"])])))
            await emit_event_once(setup, project_id, "session.completed", {"session_id": str(session.id)},
                                  dedup_key=f"planning-lock:{session.id}")
            run_id, task_id = run.id, task.id
            await setup.commit()

        provider_observations = []

        async def fake_completion(**_request):
            provider_observations.append(caller.in_nested_transaction())
            # Prove the tick's sole SQLite writer has been released while the
            # controller is in flight, rather than merely testing savepoint flags.
            async with factory() as writer:
                await writer.execute(text("BEGIN IMMEDIATE"))
                await writer.commit()
            return {"choices": [{"message": {"content": json.dumps({"decision": {
                "action_type": "noop", "reason": "Await independent plan review.", "wake_when": NOOP_WAKE_WHEN,
            }})}}]}

        class FixtureDecisionAdapter(OrchestrationDecisionAdapter):
            def __init__(self):
                super().__init__(completion_fn=fake_completion)
        monkeypatch.setattr(service_module, "OrchestrationDecisionAdapter", FixtureDecisionAdapter)

        async def no_start(*_args):
            return 0

        monkeypatch.setattr(service, "_start_released_tasks", no_start)
        async with factory() as caller:
            result = await service.tick(caller, run_id)
            assert not caller.in_nested_transaction()
        assert result["authorized_execution"]["step"] == "plan_decision"
        assert provider_observations == [False]
        async with factory() as reader:
            durable = await reader.get(OrchestrationRun, run_id)
            assert durable.plan_state.get("plan_artifact_id"), json.dumps(durable.plan_state)
            assert await reader.scalar(select(func.count()).select_from(Artifact).where(
                Artifact.linked_task_id == task_id)) == 1
            assert await reader.scalar(select(func.count()).select_from(OrchestrationDecision).where(
                OrchestrationDecision.run_id == run_id, OrchestrationDecision.validator_status == "accepted")) == 1
            assert durable.event_cursor is not None
        # Durable ingestion is replayed without another controller call or Artifact.
        async with factory() as replay:
            await service.tick(replay, run_id)
        assert provider_observations == [False]
    finally:
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("caller_savepoint", [False, True])
async def test_ingest_preserves_caller_savepoint_and_outer_rollback(test_engine, tmp_path, caller_savepoint):
    """RELEASE must never publish an Artifact ahead of its caller's transaction.

    Deliberately make ingest's INSERT the first DML in the caller transaction:
    SQLite legacy transaction control otherwise lets an outermost SAVEPOINT
    commit an Artifact even when SQLAlchemy later rolls the outer transaction back.
    """
    if test_engine.dialect.name != "sqlite":
        pytest.skip("SQLite legacy savepoint transaction regression")
    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    async with factory() as setup:
        project = Project(name="Ingestion rollback regression", workspace_path=str(tmp_path), config={})
        setup.add(project)
        await setup.flush()
        _, _, run, task, session = await _planning_run(setup, project.id)
        ids = run.goal_id, run.id, task.id, session.id
        initial_state = dict(run.plan_state)
        await setup.commit()

    async with factory() as caller:
        async with caller.begin():
            goal = await caller.get(OrchestrationGoal, ids[0])
            run = await caller.get(OrchestrationRun, ids[1])
            task = await caller.get(Task, ids[2])
            outer_savepoint = await caller.begin_nested() if caller_savepoint else None
            await ingest(caller, goal, run, task, ids[3], run.plan_state["planning_version"], _db_output())
            assert caller.get_nested_transaction() is outer_savepoint
            assert caller.in_transaction()
            await caller.rollback()
    async with factory() as reader:
        assert await reader.scalar(select(func.count()).select_from(Artifact).where(
            Artifact.linked_task_id == ids[2])) == 0
        assert (await reader.get(OrchestrationRun, ids[1])).plan_state == initial_state
