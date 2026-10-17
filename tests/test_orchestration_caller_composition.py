"""SPR230/SPR231: real deterministic operations retain supplied caller composition."""

import asyncio
import uuid

from fastapi import HTTPException
import pytest
import pytest_asyncio
from sqlalchemy import event
from sqlalchemy.ext.asyncio import async_sessionmaker

from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun
from huddleroom.models.project import Project
from huddleroom.services.orchestration_debug_service import OrchestrationDebugService
from huddleroom.services.orchestration_service import OrchestrationService


@pytest_asyncio.fixture
async def composition_scenario(test_engine, tmp_path, monkeypatch):
    from huddleroom.services import orchestration_completion

    async def forbidden_provider(**kwargs):
        raise AssertionError("deterministic caller operation reached a provider")

    monkeypatch.setattr(orchestration_completion, "orchestration_completion", forbidden_provider)
    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    async with factory() as setup:
        project = Project(name="Caller composition", description="committed", workspace_path=str(tmp_path), config={})
        deletable = Project(name="Unrelated deletion", description="keep until caller publishes", config={})
        setup.add_all([project, deletable])
        await setup.flush()
        goal = OrchestrationGoal(project_id=project.id, objective="Committed objective", status="active")
        setup.add(goal)
        await setup.flush()
        run = OrchestrationRun(goal_id=goal.id, status="running", phase="baseline", baseline_authorized=True,
                               budget_state={}, supervision_state={})
        setup.add(run)
        await setup.commit()
        yield factory, project.id, goal.id, run.id, deletable.id


def _observe_publication(db):
    calls = []
    for name in ("before_flush", "before_commit", "after_rollback"):
        event.listen(db.sync_session, name, lambda *args, name=name: calls.append(name))
    return calls


async def _load_and_stage(db, ids, *, mixed):
    project_id, goal_id, run_id, deletable_id = ids
    # Load everything before dirtying it: test preparation must not cause autoflush.
    project = await db.get(Project, project_id)
    goal = await db.get(OrchestrationGoal, goal_id)
    run = await db.get(OrchestrationRun, run_id)
    deletable = await db.get(Project, deletable_id)
    if mixed:
        await db.delete(deletable)
    project.description = "unflushed caller Project edit"
    added = None
    if mixed:
        goal.objective = "Unflushed caller objective"
        run.budget_state = {"caller_note": "unflushed run field"}
        added = Project(id=uuid.uuid4(), name="Unflushed new Project", description="caller-only", config={})
        db.add(added)
    return project, goal, run, deletable, added


@pytest.mark.asyncio
@pytest.mark.parametrize("mixed", [False, True], ids=["project", "dirty-new-deleted-run-goal"])
@pytest.mark.parametrize("publish", [False, True], ids=["caller-rollback", "caller-commit"])
async def test_real_local_tick_stages_without_flushing_caller(composition_scenario, mixed, publish):
    factory, *ids = composition_scenario
    project_id, goal_id, run_id, deletable_id = ids
    async with factory() as caller:
        project, goal, run, deletable, added = await _load_and_stage(caller, ids, mixed=mixed)
        root = caller.sync_session.get_transaction()
        dirty_before, new_before, deleted_before = set(caller.dirty), set(caller.new), set(caller.deleted)
        calls = _observe_publication(caller)
        async with asyncio.timeout(10):
            result = await OrchestrationService().tick(caller, run_id, local_only=True)
        assert result["local_only"] is True
        assert calls == [], {"events": calls, "dirty_before": len(dirty_before), "dirty_after": len(caller.dirty)}
        assert caller.sync_session.get_transaction() is root and root.is_active
        assert project.description == "unflushed caller Project edit" and caller.is_modified(project)
        assert dirty_before <= set(caller.dirty)
        assert set(caller.new) == new_before and set(caller.deleted) == deleted_before
        if mixed:
            assert goal.objective == "Unflushed caller objective" and caller.is_modified(goal)
            assert run.budget_state == {"caller_note": "unflushed run field"}
            assert caller.is_modified(run)
            assert added in caller.new and deletable in caller.deleted
        assert run.supervision_state["evaluated_at"]
        assert run.supervision_state["context_fingerprint"]
        staged_evaluation = run.supervision_state["evaluated_at"]
        async with factory() as fresh:
            assert (await fresh.get(Project, project_id)).description == "committed"
            assert (await fresh.get(OrchestrationGoal, goal_id)).objective == "Committed objective"
            persisted_run = await fresh.get(OrchestrationRun, run_id)
            assert persisted_run.budget_state == {} and persisted_run.supervision_state == {}
            assert await fresh.get(Project, deletable_id) is not None
            if added is not None:
                assert await fresh.get(Project, added.id) is None
        if publish:
            await caller.commit()  # Only the caller may publish the composed unit of work.
        else:
            await caller.rollback()
    async with factory() as fresh:
        assert (await fresh.get(Project, project_id)).description == ("unflushed caller Project edit" if publish else "committed")
        assert (await fresh.get(OrchestrationGoal, goal_id)).objective == (
            "Unflushed caller objective" if publish and mixed else "Committed objective"
        )
        persisted_run = await fresh.get(OrchestrationRun, run_id)
        assert persisted_run.budget_state == ({"caller_note": "unflushed run field"} if publish and mixed else {})
        if publish:
            assert persisted_run.supervision_state["evaluated_at"] == staged_evaluation
        else:
            assert persisted_run.supervision_state == {}
        assert (await fresh.get(Project, deletable_id) is None) is (publish and mixed)
        if added is not None:
            assert (await fresh.get(Project, added.id) is not None) is publish


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["explicit", "nested", "preflushed"])
async def test_real_debug_conflict_preserves_pending_caller_objects(composition_scenario, shape):
    factory, *ids = composition_scenario
    project_id, goal_id, run_id, deletable_id = ids
    async with factory() as caller:
        if shape != "preflushed":
            await caller.begin()
        if shape == "nested":
            await caller.begin_nested()  # Establish shape BEFORE preparing any pending edits.
        if shape == "preflushed":
            project = await caller.get(Project, project_id)
            project.description = "caller writer"
            await caller.flush()
        project, goal, run, deletable, added = await _load_and_stage(caller, ids, mixed=True)
        root, nested = caller.sync_session.get_transaction(), caller.get_nested_transaction()
        dirty_before, new_before, deleted_before = set(caller.dirty), set(caller.new), set(caller.deleted)
        calls = _observe_publication(caller)
        async with asyncio.timeout(2):
            with pytest.raises(HTTPException) as refused:
                await OrchestrationDebugService().step(caller, project_id, goal_id, "goal_definition")
        assert refused.value.status_code == 409
        assert refused.value.detail["conflict"] == "orchestration_transaction_shape"
        assert calls == []
        assert caller.sync_session.get_transaction() is root and root.is_active
        assert caller.get_nested_transaction() is nested
        assert set(caller.dirty) == dirty_before
        assert set(caller.new) == new_before and set(caller.deleted) == deleted_before
        assert project.description == "unflushed caller Project edit"
        assert goal.objective == "Unflushed caller objective"
        assert run.budget_state == {"caller_note": "unflushed run field"}
        async with factory() as fresh:
            assert (await fresh.get(Project, project_id)).description == "committed"
            assert (await fresh.get(OrchestrationRun, run_id)).supervision_state == {}
            assert await fresh.get(Project, added.id) is None
            assert await fresh.get(Project, deletable_id) is not None
        await caller.rollback()


@pytest.mark.asyncio
async def test_safe_no_pending_local_tick_stages_until_caller_rollback(composition_scenario):
    factory, _, _, run_id, _ = composition_scenario
    async with factory() as caller:
        run = await caller.get(OrchestrationRun, run_id)  # Read-only AUTOBEGIN, not a fresh owned operation.
        root = caller.sync_session.get_transaction()
        calls = _observe_publication(caller)
        async with asyncio.timeout(10):
            result = await OrchestrationService().tick(caller, run_id, local_only=True)
        assert result["local_only"] is True and run.supervision_state["evaluated_at"]
        assert caller.is_modified(run)
        assert caller.sync_session.get_transaction() is root and root.is_active
        assert calls == []
        async with factory() as fresh:
            assert (await fresh.get(OrchestrationRun, run_id)).supervision_state == {}
        await caller.rollback()
    async with factory() as fresh:
        assert (await fresh.get(OrchestrationRun, run_id)).supervision_state == {}


async def _committed_database_bytes(factory):
    """Canonical complete row bytes, not volatile SQLite file/WAL metadata."""
    import json
    from sqlalchemy import select
    from huddleroom.models.base import Base

    async with factory() as reader:
        rows = {}
        for table in Base.metadata.sorted_tables:
            result = await reader.execute(select(table))
            rows[table.name] = sorted(json.dumps(dict(row), sort_keys=True, default=str) for row in result.mappings())
        return json.dumps(rows, sort_keys=True).encode()


@pytest.mark.asyncio
async def test_no_event_unauthorized_tick_is_byte_pure(composition_scenario):
    factory, _, _, run_id, _ = composition_scenario
    async with factory() as setup:
        run = await setup.get(OrchestrationRun, run_id)
        run.baseline_authorized = False
        await setup.commit()
    before = await _committed_database_bytes(factory)
    async with factory() as caller:
        run = await caller.get(OrchestrationRun, run_id)
        root = caller.sync_session.get_transaction()
        calls = _observe_publication(caller)
        async with asyncio.timeout(10):
            result = await OrchestrationService().tick(caller, run_id)
        assert result["processed_events"] == 0
        assert calls == []
        assert caller.sync_session.get_transaction() is root and root.is_active
        assert not caller.dirty and not caller.new and not caller.deleted
        assert run.event_cursor is None and run.baseline_authorized is False
        await caller.rollback()
    assert await _committed_database_bytes(factory) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("native_state", ["missing", "unknown"])
async def test_native_proof_unavailable_refuses_without_flushing(composition_scenario, monkeypatch, native_state):
    from types import SimpleNamespace

    factory, *ids = composition_scenario
    _, _, run_id, _ = ids
    async with factory() as caller:
        project, goal, run, _, _ = await _load_and_stage(caller, ids, mixed=True)
        root = caller.sync_session.get_transaction()
        dirty, new, deleted = set(caller.dirty), set(caller.new), set(caller.deleted)
        real_connection = caller.connection

        async def connection(*args, **kwargs):
            bound = await real_connection(*args, **kwargs)
            real_raw = await bound.get_raw_connection()

            class Driver:
                def __getattr__(self, name):
                    if name == "in_transaction":
                        if native_state == "missing":
                            raise AttributeError(name)
                        return None
                    return getattr(real_raw.driver_connection, name)

            async def get_raw_connection():
                return SimpleNamespace(driver_connection=Driver())

            return SimpleNamespace(get_raw_connection=get_raw_connection)

        # Only native property availability is simulated; the safety probe and real bound session execute.
        monkeypatch.setattr(caller, "connection", connection)
        calls = _observe_publication(caller)
        async with asyncio.timeout(2):
            with pytest.raises(HTTPException) as refused:
                await OrchestrationService().tick(caller, run_id, local_only=True)
        assert refused.value.status_code == 409
        assert refused.value.detail["conflict"] == "orchestration_transaction_shape"
        assert calls == []
        assert caller.sync_session.get_transaction() is root and root.is_active
        assert set(caller.dirty) == dirty and set(caller.new) == new and set(caller.deleted) == deleted
        assert project.description == "unflushed caller Project edit" and goal.objective == "Unflushed caller objective"
        assert run.supervision_state == {} and run.budget_state == {"caller_note": "unflushed run field"}
        await caller.rollback()


@pytest.mark.asyncio
async def test_owner_is_session_task_bound_and_restores_after_error(composition_scenario):
    from huddleroom.services.orchestration_baseline_runtime import (
        baseline_operation, control_operation_owner, current_baseline_operation, owns_control_transaction,
    )

    factory, *ids = composition_scenario
    _, goal_id, run_id, _ = ids
    async with factory() as owner, factory() as foreign:
        await _load_and_stage(owner, ids, mixed=True)
        calls = _observe_publication(owner)
        assert not owns_control_transaction(owner) and current_baseline_operation() is None
        with pytest.raises(RuntimeError, match="injected owner failure"):
            async with control_operation_owner(owner):
                assert owns_control_transaction(owner) and not owns_control_transaction(foreign)
                assert await asyncio.create_task(_owner_in_child(owner)) is False
                with pytest.raises(RuntimeError, match="injected operation failure"):
                    async with baseline_operation(owner, goal_id, owns_transaction=True, run_id=run_id, register=False):
                        operation = current_baseline_operation()
                        assert operation.db is owner and operation.owner_task is asyncio.current_task()
                        assert owns_control_transaction(owner) and not owns_control_transaction(foreign)
                        assert await asyncio.create_task(_owner_in_child(owner)) is False
                        raise RuntimeError("injected operation failure")
                assert current_baseline_operation() is None and owns_control_transaction(owner)
                async with control_operation_owner(foreign):
                    assert owns_control_transaction(foreign) and not owns_control_transaction(owner)
                assert owns_control_transaction(owner) and not owns_control_transaction(foreign)
                raise RuntimeError("injected owner failure")
        assert not owns_control_transaction(owner) and not owns_control_transaction(foreign)
        assert current_baseline_operation() is None and calls == []
        await owner.rollback()


async def _owner_in_child(db):
    from huddleroom.services.orchestration_baseline_runtime import owns_control_transaction
    # No concurrent session I/O: this checks inherited ContextVar identity only.
    return owns_control_transaction(db)


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["budget", "goal_id", "both"])
async def test_committed_decision_ignores_caller_only_budget_and_goal_id(test_engine, tmp_path, monkeypatch, mutation):
    from copy import deepcopy
    from sqlalchemy import select
    from huddleroom.models.orchestration import OrchestrationAction, OrchestrationDecision
    from huddleroom.services import orchestration_service
    from tests.orchestration_wake_helpers import NOOP_WAKE_WHEN
    from tests.test_orchestration_llm_decision_adapter import _make_committed_run, RecordingAdapter

    service, factory, _, goal_id, run_id = await _make_committed_run(test_engine, tmp_path)
    monkeypatch.setattr(orchestration_service, "AsyncSessionLocal", factory)
    async with factory() as caller:
        run = await caller.get(OrchestrationRun, run_id)
        goal = await caller.get(OrchestrationGoal, goal_id)
        original_budget, original_run_budget, original_plan = deepcopy(goal.budget), deepcopy(run.budget_state), deepcopy(run.plan_state)
        foreign_goal_id = uuid.uuid4()
        if mutation in {"budget", "both"}:
            goal.budget = {"llm_calls": 0}
            run.budget_state = {"llm_calls": 999999}
            run.plan_state = {"budget_required": True}
        if mutation in {"goal_id", "both"}:
            run.goal_id = foreign_goal_id  # Never committed; cannot authorize/admit against this target.
        root = caller.sync_session.get_transaction()
        dirty_before = set(caller.dirty)
        calls = _observe_publication(caller)
        adapter = RecordingAdapter({"action_type": "noop", "reason": "Committed authority", "wake_when": NOOP_WAKE_WHEN})
        async with asyncio.timeout(10):
            decision = await service.request_llm_decision(caller, run_id, adapter=adapter)
        assert calls == [] and caller.sync_session.get_transaction() is root and root.is_active
        assert dirty_before <= set(caller.dirty)
        assert len(adapter.contexts) == 1
        context = adapter.contexts[0]
        assert context["goal"]["id"] == str(goal_id) and context["run"]["goal_id"] == str(goal_id)
        assert context["goal"]["budget"] == original_budget
        assert context["run"]["budget_state"] == original_run_budget
        assert context["run"]["plan_state"] == original_plan
        assert decision.validator_status == "accepted"
        if mutation in {"budget", "both"}:
            assert goal.budget == {"llm_calls": 0} and run.plan_state == {"budget_required": True}
            assert run.budget_state == {"llm_calls": 999999}
        if mutation in {"goal_id", "both"}:
            assert run.goal_id == foreign_goal_id
        decision_id = decision.id
        await caller.rollback()
    async with factory() as fresh:
        assert (await fresh.get(OrchestrationRun, run_id)).goal_id == goal_id
        assert (await fresh.get(OrchestrationGoal, goal_id)).budget == original_budget
        assert (await fresh.get(OrchestrationDecision, decision_id)).validator_status == "accepted"
        admissions = list(await fresh.scalars(select(OrchestrationAction).where(
            OrchestrationAction.run_id == run_id, OrchestrationAction.action_type == "admit_control_call",
        )))
        assert len(admissions) == 1 and admissions[0].dispatch_contract["provider_completed"] is True


@pytest.mark.asyncio
async def test_pending_steering_refuses_local_tick_before_any_caller_mutation(composition_scenario, monkeypatch):
    from copy import deepcopy
    from sqlalchemy import select
    from huddleroom.models.orchestration_steering import (
        OrchestrationSteeringRequest, OrchestrationSteeringState, OrchestrationSteeringTransition,
        steering_request_id, steering_state_id,
    )
    from huddleroom.services.orchestration_steering import OrchestrationSteeringService
    from huddleroom.config import settings
    from huddleroom.dependencies import _ANON_USER

    factory, *ids = composition_scenario
    _, goal_id, run_id, _ = ids
    monkeypatch.setattr(settings, "orchestration_conversation_steering_enabled", True)
    async with factory() as setup:
        goal = await setup.get(OrchestrationGoal, goal_id)
        run = await setup.get(OrchestrationRun, run_id)
        run.phase = "authorized"
        client_id = uuid.uuid4()
        request_id = steering_request_id(goal_id, _ANON_USER.id, client_id)
        setup.add(OrchestrationSteeringState(id=steering_state_id(goal_id), goal_id=goal_id, inbox_version=1, direction_version=0))
        setup.add(OrchestrationSteeringRequest(
            id=request_id, goal_id=goal_id, actor_id=_ANON_USER.id, client_request_id=client_id,
            sequence=1, submitted_run_id=run_id, directive="Keep the requested change narrowly scoped.",
            target_type="goal", target_id=str(goal_id), scope="run", lifetime="remaining_current_run",
            impact_summary="Limit the current work", status="pending", reason_code="submitted",
            contract_version=OrchestrationSteeringService._contract_version(goal),
            plan_version=OrchestrationSteeringService._plan_version(run),
        ))
        await setup.commit()
    before = await _committed_database_bytes(factory)
    async with factory() as caller:
        project, goal, run, _, _ = await _load_and_stage(caller, ids, mixed=True)
        # Do not load a new ORM identity after pending edits without no_autoflush.
        with caller.no_autoflush:
            request = await caller.get(OrchestrationSteeringRequest, request_id)
        root = caller.sync_session.get_transaction()
        dirty, new, deleted = set(caller.dirty), set(caller.new), set(caller.deleted)
        original_supervision = deepcopy(run.supervision_state)
        calls = _observe_publication(caller)
        async with asyncio.timeout(2):
            with pytest.raises(HTTPException) as refused:
                await OrchestrationService().tick(caller, run_id, local_only=True)
        assert refused.value.status_code == 409
        assert calls == [] and caller.sync_session.get_transaction() is root and root.is_active
        assert set(caller.dirty) == dirty and set(caller.new) == new and set(caller.deleted) == deleted
        assert project.description == "unflushed caller Project edit" and goal.objective == "Unflushed caller objective"
        assert run.budget_state == {"caller_note": "unflushed run field"} and run.supervision_state == original_supervision
        assert request.status == "pending" and request.considered_at is None
        with caller.no_autoflush:
            assert not list(await caller.scalars(select(OrchestrationSteeringTransition).where(
                OrchestrationSteeringTransition.request_id == request_id,
            )))
        await caller.rollback()
    assert await _committed_database_bytes(factory) == before


def _spy_service_publication(db):
    """Record explicit AsyncSession.commit/rollback calls (SAVEPOINT release also fires before_commit, so events can't tell)."""
    calls = []
    for name in ("commit", "rollback"):
        original = getattr(db, name)

        async def spy(*, original=original, name=name):
            calls.append(name)
            return await original()
        setattr(db, name, spy)
    return calls


@pytest.mark.asyncio
@pytest.mark.usefixtures("safe_goal_analysis")
@pytest.mark.parametrize("publish", [False, True], ids=["caller-rollback", "caller-commit"])
async def test_real_debug_step_safe_clean_caller_composes_without_service_publication(composition_scenario, publish):
    """SPR230/231: real step() on a clean read-only AUTOBEGIN caller proceeds; only the caller publishes."""
    from huddleroom.models.orchestration_process import OrchestrationProcessRun
    from sqlalchemy import func, select

    factory, project_id, goal_id, run_id, _ = composition_scenario

    async def process_rows(session):
        return await session.scalar(select(func.count()).select_from(OrchestrationProcessRun).where(
            OrchestrationProcessRun.goal_id == goal_id))

    async with factory() as caller:
        await caller.get(OrchestrationRun, run_id)  # Read-only AUTOBEGIN: supplied caller, not an owned operation.
        root = caller.sync_session.get_transaction()
        assert root is not None and root.is_active
        calls = _spy_service_publication(caller)
        async with asyncio.timeout(10):
            result = await OrchestrationDebugService().step(caller, project_id, goal_id, "goal_definition")
        assert result["process_type"] == "goal_definition" and result["run_id"] == run_id
        assert calls == [], calls  # zero service commit/rollback
        assert caller.sync_session.get_transaction() is root and root.is_active
        assert await process_rows(caller) >= 1  # staged inside the caller's transaction
        async with factory() as fresh:
            assert await process_rows(fresh) == 0  # nothing visible before the caller publishes
        if publish:
            await caller.commit()
        else:
            await caller.rollback()
    async with factory() as fresh:
        assert (await process_rows(fresh) >= 1) is publish


@pytest.mark.asyncio
@pytest.mark.usefixtures("safe_goal_analysis")
async def test_real_debug_step_safe_caller_does_not_flush_unrelated_pending_edit(composition_scenario):
    """Strict contract (plan row 3): a safe caller with an unrelated pending edit gets a retryable 409; nothing flushes."""
    factory, project_id, goal_id, run_id, _ = composition_scenario
    async with factory() as caller:
        project = await caller.get(Project, project_id)
        await caller.get(OrchestrationRun, run_id)
        project.description = "unflushed caller Project edit"  # unrelated to the debug step's own writes
        root = caller.sync_session.get_transaction()
        calls = _spy_service_publication(caller)
        flushes = _observe_publication(caller)
        async with asyncio.timeout(10):
            with pytest.raises(HTTPException) as refused:
                await OrchestrationDebugService().step(caller, project_id, goal_id, "goal_definition")
        assert refused.value.status_code == 409
        assert refused.value.detail["conflict"] == "orchestration_transaction_shape"
        assert calls == [], calls
        assert caller.sync_session.get_transaction() is root and root.is_active
        # Strict, unweakened: the unrelated edit must stay pending (never flushed by the service).
        assert "before_flush" not in flushes, flushes
        assert caller.is_modified(project) and project in caller.dirty
        assert not any(type(o).__name__ == "OrchestrationProcessRun" for o in caller.new)
        async with factory() as fresh:
            assert (await fresh.get(Project, project_id)).description == "committed"
            from sqlalchemy import func, select
            from huddleroom.models.orchestration_process import OrchestrationProcessRun
            assert await fresh.scalar(select(func.count()).select_from(OrchestrationProcessRun).where(
                OrchestrationProcessRun.goal_id == goal_id)) == 0  # no effect staged
        await caller.rollback()
