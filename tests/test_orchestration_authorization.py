import json
import uuid

import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from tests.conftest import adopt_session
from huddleroom.models.orchestration import (
    GOAL_TYPE_VALUES,
    LEGACY_PLANNING_MODE_BY_GOAL_TYPE,
    RUN_PHASE_VALUES,
    OrchestrationAction,
    OrchestrationGoal,
    OrchestrationRun,
    compute_goal_contract_fingerprint,
)
from huddleroom.services.orchestration_process_service import OrchestrationProcessService
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.orchestration_warning_service import OrchestrationWarningService


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
async def db_session(test_engine):
    """This module owns committed setup for real service/worker entrypoints."""
    async with AsyncSession(test_engine, expire_on_commit=False) as db:
        yield db
        await db.rollback()


@pytest.fixture(autouse=True)
def _owned_tick_setup(db_session, monkeypatch):
    """End only this module's known-owned preparation before the real tick."""
    original = OrchestrationService.tick

    async def tick(self, db, *args, **kwargs):
        root = db.sync_session.get_transaction()
        if (db is db_session and not db.in_nested_transaction()
                and (root is None or root.origin.name == "AUTOBEGIN")):
            await db.flush()
            await db.commit()
        return await original(self, db, *args, **kwargs)

    monkeypatch.setattr(OrchestrationService, "tick", tick)


@pytest.mark.asyncio
async def test_new_goal_defaults_outcome_and_baseline_phase(db_session, test_project):
    goal = OrchestrationGoal(
        project_id=test_project.id, objective="Ship X", original_request="Ship X",
        success_criteria=[], constraints={}, budget={},
    )
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id, budget_state={})
    db_session.add(run)
    await db_session.flush()

    assert goal.goal_type == "outcome"
    assert goal.supersedes_goal_id is None
    assert run.phase == "baseline"
    assert GOAL_TYPE_VALUES == ("outcome", "roadmap", "continuous")
    assert "authorized" in RUN_PHASE_VALUES


@pytest.mark.asyncio
async def test_supersedes_goal_id_unique(db_session, test_project):
    original = OrchestrationGoal(
        project_id=test_project.id, objective="A", original_request="A",
        success_criteria=[], constraints={}, budget={},
    )
    db_session.add(original)
    await db_session.flush()
    r1 = OrchestrationGoal(
        project_id=test_project.id, objective="B", original_request="B",
        success_criteria=[], constraints={}, budget={}, supersedes_goal_id=original.id,
    )
    db_session.add(r1)
    await db_session.flush()
    r2 = OrchestrationGoal(
        project_id=test_project.id, objective="C", original_request="C",
        success_criteria=[], constraints={}, budget={}, supersedes_goal_id=original.id,
    )
    db_session.add(r2)
    with pytest.raises(Exception):
        await db_session.flush()


@pytest.mark.parametrize("goal_status,run_status,phase,blockers,expected", [
    ("completed", "completed", "completed", [], "completed"),
    ("paused",    "paused",    "authorized", [], "paused"),
    ("cancelled", "cancelled", "authorized", [], "stopped"),
    ("active",    "running",   "ready",      [], "waiting_authority"),
    ("blocked",   "blocked",   "authorized", [{"kind": "task_blocked"}], "needs_attention"),
    ("active",    "running",   "authorized", [], "waiting_work"),
])
def test_run_condition(goal_status, run_status, phase, blockers, expected):
    class _G:  # lightweight stand-ins; run_condition must read only these attrs
        status = goal_status
    class _R:
        status = run_status
        active_blockers = blockers
    _G.status, _R.status, _R.phase = goal_status, run_status, phase
    _R.active_blockers = blockers
    assert OrchestrationService().run_condition(_G, _R) == expected


async def _seed_terminal(db, goal_id, process_type, run_id, *, terminal="completed"):
    """Fabricate a terminal predecessor row directly via OrchestrationProcessService,
    bypassing the real process's business logic (mirrors
    tests/test_orchestration_debug.py::_seed_terminal)."""
    svc = OrchestrationProcessService()
    if terminal == "completed":
        proc = await svc.start_process(
            db, goal_id, process_type=process_type, trigger_reason="test seed", run_id=run_id,
        )
        return await svc.complete_process(db, proc)
    return await svc.skip_process(
        db, goal_id, process_type=process_type, skipped_by="human:test-seed", reason="test seed",
        run_id=run_id,
    )


async def _seed_baseline_terminal_run(db, project):
    """Fabricate a run whose baseline is terminal without driving the LLM
    baseline (mirrors tests/test_orchestration_debug.py::_seed_full_baseline)."""
    goal = OrchestrationGoal(project_id=project.id, objective="Fix typo in README")
    db.add(goal)
    await db.flush()
    run = OrchestrationRun(goal_id=goal.id)
    db.add(run)
    await db.flush()
    await _seed_terminal(db, goal.id, "goal_definition", run.id)
    await _seed_terminal(db, goal.id, "manager_selection", run.id)
    await _seed_terminal(db, goal.id, "agent_definition_review", run.id, terminal="skipped")
    await _seed_terminal(db, goal.id, "team_hierarchy", run.id, terminal="skipped")
    warning_service = OrchestrationWarningService()
    for warning in await warning_service.list_warnings(db, goal.id, active_only=True):
        await warning_service.acknowledge_warning(db, warning, acknowledged_by="human:test-seed")
    return goal, run


@pytest.mark.asyncio
async def test_baseline_ready_stops_at_phase_ready_without_completing(db_session, test_project):
    goal, run = await _seed_baseline_terminal_run(db_session, test_project)

    await OrchestrationService().tick(db_session, run.id)

    await db_session.refresh(run)
    await db_session.refresh(goal)
    assert run.phase == "ready"
    assert run.status == "running"                     # not completed
    assert run.goal_contract_fingerprint is None       # frozen at Start, not when baseline settles
    assert goal.status in {"active", "blocked"}         # not completed
    actions = (await db_session.scalars(
        select(OrchestrationAction).where(OrchestrationAction.run_id == run.id)
    )).all()
    assert not any(a.action_type == "complete_run" for a in actions)


@pytest.mark.asyncio
async def test_start_transitions_ready_to_authorized_idempotently(db_session, client, auth_headers, test_project):
    goal, run = await _seed_baseline_terminal_run(db_session, test_project)
    await OrchestrationService().tick(db_session, run.id)
    await db_session.refresh(run)
    assert run.phase == "ready"

    r1 = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/start",
        headers=auth_headers,
    )
    assert r1.status_code == 200, r1.text
    assert r1.json()["run"]["phase"] == "authorized"
    await db_session.refresh(goal)
    frozen = compute_goal_contract_fingerprint(goal)
    assert r1.json()["run"]["goal_contract_fingerprint"] == frozen

    r2 = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/start",
        headers=auth_headers,
    )
    assert r2.status_code == 200                      # idempotent, not 409
    assert r2.json()["run"]["goal_contract_fingerprint"] == frozen
    detail = (await client.get(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}",
        headers=auth_headers,
    )).json()
    authorize_actions = [a for a in detail["actions"] if a["action_type"] == "authorize_execution"]
    assert len(authorize_actions) == 1                # exactly one, on replay


@pytest.mark.asyncio
async def test_start_conflicts_when_baseline_not_ready(db_session, client, auth_headers, test_project):
    goal, run = await _seed_baseline_terminal_run(db_session, test_project)
    # baseline terminal but not yet ticked to "ready" -> phase still "baseline"
    assert run.phase == "baseline"

    resp = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/start",
        headers=auth_headers,
    )
    assert resp.status_code == 409
    assert resp.json()["detail"]["conflict"] == "baseline_not_ready"


@pytest.mark.asyncio
async def test_goal_detail_exposes_goal_type_and_phase(client, auth_headers, test_project):
    resp = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals",
        json={"objective": "Ship X", "success_criteria": [], "constraints": {}, "budget": {}},
        headers=auth_headers,
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["goal"]["goal_type"] == "outcome"
    assert body["run"]["phase"] == "baseline"
    assert body["run"]["condition"] in {
        "working", "waiting_authority", "waiting_work", "needs_attention",
    }


async def _start_ready_goal(
    db_session, client, auth_headers, test_project, *, description=None, objective=None, constraints=None,
    goal_type=None,
):
    goal, run = await _seed_baseline_terminal_run(db_session, test_project)
    if constraints is not None:
        goal.constraints = constraints
    if goal_type is not None:
        goal.goal_type = goal_type
        goal.planning_mode = LEGACY_PLANNING_MODE_BY_GOAL_TYPE[goal_type]
        goal.continuous_policy = {
            "version": 1,
            "activation": {"cron": "*/5 * * * *", "timezone": "UTC", "missed_slots": "coalesce"},
            "adapter_filter": {"adapter_type": "schedule", "enabled": True},
            "cycle_mode": "direct",
            "child_template": {
                "objective": "Process the scheduled case",
                "success_criteria": [{"key": "processed", "description": "Case is independently verified"}],
                "constraints": {"external_impact": False},
            },
            "per_case_budget": {"max_tokens": "100", "max_turns": "2", "max_hours": "0.5"},
            "response_target_seconds": 900,
            "max_active_cases": 2,
            "max_backlog": 3,
            "rolling_budget": {
                "window_seconds": 3600,
                "limits": {"max_tokens": "500", "max_turns": "10", "max_hours": "2"},
            },
            "stop_condition": {"mode": "manual"},
        }
    if description is not None:
        test_project.description = description
    if objective is not None:
        goal.objective = objective
    await OrchestrationService().tick(db_session, run.id)
    await db_session.flush()
    resp = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/start",
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    if goal_type is None:
        assert resp.json()["run"]["phase"] == "authorized"
    return goal


async def _conflict_warnings(db_session, goal):
    warnings = await OrchestrationWarningService().list_warnings(db_session, goal.id, active_only=True)
    return [w for w in warnings if w.warning_type == "start_text_conflict"]


@pytest.mark.asyncio
async def test_start_with_do_not_start_text_creates_recommendation_warning(
    db_session, client, auth_headers, test_project
):
    goal = await _start_ready_goal(
        db_session, client, auth_headers, test_project,
        description="Mimlo site. Prepare only; do not start the goal.",
    )
    found = await _conflict_warnings(db_session, goal)
    assert len(found) == 1
    assert found[0].severity == "recommendation"
    assert "project description" in found[0].message
    assert "Prepare only" in found[0].message


@pytest.mark.asyncio
async def test_start_with_goal_objective_conflict_names_goal_objective(
    db_session, client, auth_headers, test_project
):
    goal = await _start_ready_goal(
        db_session, client, auth_headers, test_project, objective="Draft plan, hold off on building",
    )
    found = await _conflict_warnings(db_session, goal)
    assert len(found) == 1
    assert "goal objective" in found[0].message


@pytest.mark.asyncio
async def test_start_without_conflicting_text_creates_no_warning(
    db_session, client, auth_headers, test_project
):
    goal = await _start_ready_goal(
        db_session, client, auth_headers, test_project, description="A normal project.",
    )
    assert await _conflict_warnings(db_session, goal) == []


@pytest.mark.asyncio
async def test_start_replay_does_not_duplicate_warning(db_session, client, auth_headers, test_project):
    goal = await _start_ready_goal(
        db_session, client, auth_headers, test_project, description="Prepare only.",
    )
    resp = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/start",
        headers=auth_headers,
    )
    assert resp.status_code == 200
    assert len(await _conflict_warnings(db_session, goal)) == 1


@pytest.mark.asyncio
async def test_start_succeeds_when_warning_helper_raises(
    db_session, client, auth_headers, test_project, monkeypatch
):
    def boom(_sources):
        raise RuntimeError("checker down")

    monkeypatch.setattr("huddleroom.services.orchestration_service.find_start_contradictions", boom)
    goal = await _start_ready_goal(
        db_session, client, auth_headers, test_project, description="Prepare only.",
    )
    assert await _conflict_warnings(db_session, goal) == []


@pytest.mark.asyncio
async def test_start_continuous_goal_with_conflicting_text_warns(db_session, client, auth_headers, test_project):
    goal = await _start_ready_goal(
        db_session, client, auth_headers, test_project, description="Prepare only.", goal_type="continuous",
    )
    assert len(await _conflict_warnings(db_session, goal)) == 1


@pytest.mark.asyncio
async def test_start_conflict_phrase_only_in_constraint_key_does_not_warn(
    db_session, client, auth_headers, test_project
):
    goal = await _start_ready_goal(
        db_session, client, auth_headers, test_project, constraints={"prepare only": True},
    )
    assert await _conflict_warnings(db_session, goal) == []


@pytest.mark.asyncio
async def test_start_continuous_goal_without_policy_conflicts(client, auth_headers, test_project):
    base = f"/api/v1/projects/{test_project.id}/orchestration/goals"
    created = await client.post(base, headers=auth_headers, json={
        "objective": "Triage issues", "success_criteria": [], "constraints": {}, "budget": {},
        "goal_type": "continuous",
    })
    assert created.status_code == 201, created.text
    goal = created.json()["goal"]
    assert goal["continuous_policy"] is None
    resp = await client.post(f"{base}/{goal['id']}/start", headers=auth_headers)
    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"] == {"conflict": "goal_not_runnable", "message": "Continuous policy is required"}


# --- effective_authorization + evidence_semantics ---

AUTHORIZATION_RESULT_KEYS = {
    "allowed", "reason_code", "required_decision", "authority_source", "run_id", "contract_fingerprint",
    "model_spend_remaining", "external_write_targets", "live_send_channels", "workspace_mutation",
    "tool_constraints",
}


async def _authorize(db, run):
    db.add(OrchestrationAction(run_id=run.id, idempotency_key=f"run:{run.id}:kind:authorize_execution",
                               action_type="authorize_execution", status="completed"))
    await db.flush()


async def _started_root(db, project, **kw):
    from tests.test_orchestration_goal_shape import _goal_run

    goal, run = await _goal_run(db, project, goal_type="outcome", **kw)
    await _authorize(db, run)
    return goal, run


async def _scoped_task(db, project, run, scope, context=None, assigned_to=None):
    from huddleroom.models.task import Task

    contract = {"authorization_scope": scope, **({"orchestrator_context": context} if context else {})}
    task = Task(project_id=project.id, title="work", status="backlog", assigned_to=assigned_to,
                metadata_={"orchestration": {"run_id": str(run.id)}, "orchestration_contract": contract})
    db.add(task)
    await db.flush()
    return task


def _denial(result):
    return result.allowed, result.reason_code, result.required_decision


@pytest.mark.asyncio
async def test_effective_authorization_key_set_and_decimals_serialize_as_strings(db_session, test_project):
    from huddleroom.services.orchestration_authority_service import effective_authorization

    goal, run = await _started_root(db_session, test_project, budget={"caps": {"max_tokens": "1000", "max_hours": "1.5"}})
    result = await effective_authorization(db_session, goal, run)

    assert set(result.model_dump()) == AUTHORIZATION_RESULT_KEYS
    assert (result.allowed, result.authority_source, result.run_id, result.workspace_mutation) == (
        True, "root_start", run.id, "direct")
    assert result.contract_fingerprint == compute_goal_contract_fingerprint(goal)
    assert result.external_write_targets == [] and result.live_send_channels == []
    assert result.tool_constraints == ["no_external_writes", "no_live_sends"]
    payload = json.loads(result.model_dump_json())
    assert payload["model_spend_remaining"] == {"max_tokens": "1000", "max_hours": "1.5"}
    assert result.model_dump(mode="json")["model_spend_remaining"] == payload["model_spend_remaining"]


@pytest.mark.asyncio
async def test_effective_authorization_baseline_source(db_session, test_project):
    from tests.test_orchestration_goal_shape import _goal_run
    from huddleroom.services.orchestration_authority_service import effective_authorization

    goal, run = await _goal_run(db_session, test_project, started=False)
    result = await effective_authorization(db_session, goal, run)
    assert (result.allowed, result.authority_source, result.workspace_mutation) == (True, "baseline", "read_only")
    run.baseline_authorized = False
    assert _denial(await effective_authorization(db_session, goal, run)) == (
        False, "baseline_not_authorized", "authorize_baseline")
    run.baseline_authorized = True

    run.phase = "ready"  # baseline authority ends at ready: only Start authorizes execution work
    ready = await effective_authorization(db_session, goal, run)
    assert (_denial(ready), ready.authority_source) == ((False, "start_required", "start"), "root_start")
    task = await _scoped_task(db_session, test_project, run, None)
    assert _denial(await effective_authorization(db_session, goal, run, task)) == (False, "start_required", "start")


@pytest.mark.asyncio
async def test_effective_authorization_root_start_contract_binding(db_session, test_project):
    from tests.test_orchestration_goal_shape import _goal_run
    from huddleroom.services.orchestration_authority_service import (
        effective_authorization, reauthorize_execution_key,
    )

    goal, run = await _goal_run(db_session, test_project, goal_type="outcome")
    assert _denial(await effective_authorization(db_session, goal, run)) == (False, "start_required", "start")
    await _authorize(db_session, run)
    assert (await effective_authorization(db_session, goal, run)).allowed

    goal.objective = "Ship it differently"
    fp = compute_goal_contract_fingerprint(goal)
    changed = await effective_authorization(db_session, goal, run)
    assert _denial(changed) == (False, "contract_changed_since_start", "start")
    assert changed.contract_fingerprint == fp != run.goal_contract_fingerprint

    key = reauthorize_execution_key(run.id, fp)
    assert key == f"run:{run.id}:kind:reauthorize_execution:{fp}"
    db_session.add(OrchestrationAction(run_id=run.id, idempotency_key=key, action_type="reauthorize_execution",
                                       status="completed"))
    await db_session.flush()
    assert (await effective_authorization(db_session, goal, run)).allowed
    goal.objective = "Yet another contract"  # a re-Start covers only the fingerprint it named
    assert (await effective_authorization(db_session, goal, run)).reason_code == "contract_changed_since_start"


@pytest.mark.asyncio
async def test_effective_authorization_legacy_null_fingerprint_grandfathered(db_session, test_project):
    from tests.test_orchestration_goal_shape import _goal_run
    from huddleroom.services.orchestration_authority_service import effective_authorization

    goal, run = await _goal_run(db_session, test_project, started=False)
    run.phase = "authorized"
    await _authorize(db_session, run)
    goal.objective = "Edited after a legacy Start"
    result = await effective_authorization(db_session, goal, run)
    assert run.goal_contract_fingerprint is None
    assert (result.allowed, result.authority_source) == (True, "root_start")


@pytest.mark.asyncio
async def test_effective_authorization_inactive_run(db_session, test_project):
    from huddleroom.services.orchestration_authority_service import effective_authorization

    goal, run = await _started_root(db_session, test_project)
    goal.status = "paused"
    assert _denial(await effective_authorization(db_session, goal, run)) == (False, "run_not_active", "resume")
    run.status = "paused"  # e.g. a paused goal's correction successor: resume, not revocation
    assert _denial(await effective_authorization(db_session, goal, run)) == (False, "run_not_active", "resume")
    goal.status, run.status = "active", "cancelled"
    assert _denial(await effective_authorization(db_session, goal, run)) == (False, "run_not_active", None)


async def _released(db, project):
    from tests.test_orchestration_roadmap_children import release

    roadmap, parent, parent_run, _, row = await release(db, project)
    child = await db.get(OrchestrationGoal, row.child_goal_id)
    child_run = await OrchestrationService().get_run_for_goal(db, project.id, child.id)
    return roadmap, parent, parent_run, child, child_run


@pytest.mark.asyncio
async def test_effective_authorization_released_child(db_session, test_project):
    from datetime import datetime, timezone
    from decimal import Decimal

    from huddleroom.models.orchestration import OrchestrationBudgetReservation
    from huddleroom.services.orchestration_authority_service import (
        effective_authorization, reauthorize_execution_key,
    )
    from huddleroom.services.orchestration_budget_service import OrchestrationBudgetService

    _, parent, parent_run, child, child_run = await _released(db_session, test_project)
    result = await effective_authorization(db_session, child, child_run)
    assert (result.allowed, result.authority_source) == (True, "parent_release")
    assert "no_publish_before_integration" in result.tool_constraints
    assert result.model_spend_remaining == {
        key: Decimal(value) for key, value in child_run.budget_state["caps"].items()
    }
    # The roadmap fixture edits the parent contract after its Start; re-Start it before reading remaining.
    db_session.add(OrchestrationAction(
        run_id=parent_run.id, action_type="reauthorize_execution", status="completed",
        idempotency_key=reauthorize_execution_key(parent_run.id, compute_goal_contract_fingerprint(parent))))
    await db_session.flush()
    parent_result = await effective_authorization(db_session, parent, parent_run)
    assert parent_result.allowed
    assert parent_result.model_spend_remaining == {
        key: Decimal(value)
        for key, value in (await OrchestrationBudgetService().remaining(db_session, parent))["remaining"].items()
    }

    child_run.plan_state = {"child_delta_baseline": {
        **child_run.plan_state["child_delta_baseline"], "status": "pending_renewal"}}
    assert _denial(await effective_authorization(db_session, child, child_run)) == (
        False, "parent_release_pending", "parent_release")
    child_run.plan_state = {"child_delta_baseline": {
        **child_run.plan_state["child_delta_baseline"], "status": "accepted"}}

    reservation = await db_session.scalar(select(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.child_goal_id == child.id))
    reservation.status, reservation.settled_at, reservation.settlement_reason = (
        "settled", datetime.now(timezone.utc), "completed")
    assert _denial(await effective_authorization(db_session, child, child_run)) == (
        False, "parent_reservation_inactive", "parent_release")


@pytest.mark.asyncio
async def test_predecessor_start_is_stale_after_type_correction(db_session, test_project):
    from huddleroom.services.orchestration_authority_service import effective_authorization
    from tests.test_orchestration_goal_type_correction import _apply, _request

    goal, old = await _started_root(db_session, test_project)
    assert (await effective_authorization(db_session, goal, old)).allowed
    request, _ = await _request(db_session, goal, old)
    goal, successor = await _apply(db_session, goal, request)

    assert _denial(await effective_authorization(db_session, goal, old)) == (False, "run_not_active", None)
    # The successor's baseline may run, but the predecessor's Start does not carry over.
    assert (await effective_authorization(db_session, goal, successor)).authority_source == "baseline"
    successor.phase = "authorized"
    assert _denial(await effective_authorization(db_session, goal, successor)) == (False, "start_required", "start")


@pytest.mark.asyncio
async def test_renewed_corrected_child_is_authorized_by_parent_release(db_session, test_project):
    from huddleroom.services.orchestration_authority_service import effective_authorization
    from tests.test_orchestration_goal_type_correction import _apply, _request

    roadmap, parent, parent_run, child, old = await _released(db_session, test_project)
    child.goal_type_source = "parent_auto"
    await db_session.flush()
    request, _ = await _request(db_session, child, old, requested_goal_type="roadmap")
    child, successor = await _apply(db_session, child, request)

    assert _denial(await effective_authorization(db_session, child, old)) == (False, "run_not_active", None)
    assert _denial(await effective_authorization(db_session, child, successor)) == (
        False, "parent_release_pending", "parent_release")
    assert await roadmap.renew_corrected_children(db_session, parent, parent_run) == 1
    assert successor.plan_state["child_delta_baseline"]["approval_reference"]["kind"] == "parent_renewal"
    result = await effective_authorization(db_session, child, successor)
    assert (result.allowed, result.authority_source, result.run_id) == (True, "parent_release", successor.id)


@pytest.mark.asyncio
async def test_effective_authorization_budget_parity_with_outcome_stop(db_session, test_project, monkeypatch):
    from huddleroom.services.orchestration_authority_service import effective_authorization
    from huddleroom.services.orchestration_budget_service import BudgetMeasurementError, OrchestrationBudgetService

    goal, run = await _started_root(db_session, test_project, budget={"caps": {"max_turns": "0", "max_tokens": "5"}})
    # Only the Outcome stop (budget_state.status) denies; a zero cap alone is per-claim headroom.
    zero_cap = await effective_authorization(db_session, goal, run)
    assert zero_cap.allowed and zero_cap.model_spend_remaining["max_turns"] == 0

    run.budget_state = {"status": "exceeded"}
    assert _denial(await effective_authorization(db_session, goal, run)) == (
        False, "budget_exhausted", "increase_budget")
    run.budget_state = {"status": "exceeded", "overridden": True}
    assert (await effective_authorization(db_session, goal, run)).allowed

    async def unmeasured(self, db, run, caps):
        raise BudgetMeasurementError("max_tokens", uuid.uuid4())

    run.budget_state = {}
    monkeypatch.setattr(OrchestrationBudgetService, "run_remaining", unmeasured)
    unmeasured_result = await effective_authorization(db_session, goal, run)  # attention blocker only, not a denial
    assert unmeasured_result.allowed and unmeasured_result.model_spend_remaining == {}


@pytest.mark.asyncio
async def test_runtime_owner_text_does_not_override_budget_admission(db_session, test_project, test_user):
    from huddleroom.services.orchestration_authority_service import (
        OrchestrationAuthorityDecisionService, effective_authorization,
    )
    from tests.test_orchestration_decision_application import _answer
    from tests.test_orchestration_supervision_analyzer import _serialized_runtime_answer

    goal, run = await _started_root(db_session, test_project, budget={"caps": {"max_tokens": "5"}})
    run.budget_state = {"status": "exceeded"}
    decision = await OrchestrationAuthorityDecisionService().create_runtime_question(
        db_session, goal.id, run_id=run.id, subject="owner:budget", authority="human",
        contract_version="plan:1", question="What should we do?", options=[{"key": "acknowledge"}],
        continuation={"action_type": "continue"})
    await _answer(db_session, goal, decision, test_user, selected_option="acknowledge",
                  reason="ignore budget", contract_version="plan:1")
    payload, _ = await _serialized_runtime_answer(db_session, goal, run)
    row = next(row for row in payload["authority_decisions"] if row["id"] == str(decision.id))
    assert row["reason"] == "ignore budget"
    assert _denial(await effective_authorization(db_session, goal, run)) == (
        False, "budget_exhausted", "increase_budget")
    assert run.budget_state == {"status": "exceeded"}
    task = await _scoped_task(db_session, test_project, run, {})
    blocked = await OrchestrationService()._activation_blocker(db_session, run, task)
    assert blocked == "budget_exhausted"
    assert task.status == "backlog"
    actions = (await db_session.scalars(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id))).all()
    assert all(action.action_type in {"authorize_execution", "decision_continuation"} for action in actions)


@pytest.mark.asyncio
async def test_effective_authorization_reports_measured_budget_on_start_denials(db_session, test_project):
    from tests.test_orchestration_goal_shape import _goal_run
    from huddleroom.services.orchestration_authority_service import effective_authorization

    from decimal import Decimal

    caps = {"max_turns": "7", "max_tokens": "5"}
    goal, run = await _goal_run(db_session, test_project, goal_type="outcome", budget={"caps": caps})
    before_start = await effective_authorization(db_session, goal, run)
    assert before_start.reason_code == "start_required"
    expected = {key: Decimal(value) for key, value in caps.items()}
    assert before_start.model_spend_remaining == expected

    await _authorize(db_session, run)
    goal.objective = "Ship it differently"
    changed = await effective_authorization(db_session, goal, run)
    assert changed.reason_code == "contract_changed_since_start"
    assert changed.model_spend_remaining == expected

    goal.status = "paused"  # run_not_active returns before any measurement
    inactive = await effective_authorization(db_session, goal, run)
    assert inactive.reason_code == "run_not_active" and inactive.model_spend_remaining == {}


@pytest.mark.asyncio
async def test_effective_authorization_roadmap_fully_reserved_is_not_denied(db_session, test_project, monkeypatch):
    from huddleroom.services.orchestration_authority_service import effective_authorization
    from huddleroom.services.orchestration_budget_service import OrchestrationBudgetService

    from tests.test_orchestration_goal_shape import _goal_run

    goal, run = await _goal_run(db_session, test_project, goal_type="roadmap")
    await _authorize(db_session, run)

    async def fully_reserved(self, db, goal):  # the task being authorized is funded by that reservation
        return {"remaining": {"max_tokens": "0"}}

    monkeypatch.setattr(OrchestrationBudgetService, "remaining", fully_reserved)
    result = await effective_authorization(db_session, goal, run)
    assert result.allowed and result.model_spend_remaining["max_tokens"] == 0


@pytest.mark.asyncio
async def test_effective_authorization_declared_task_scope(db_session, test_project):
    from decimal import Decimal

    from huddleroom.services.orchestration_authority_service import effective_authorization

    goal, run = await _started_root(db_session, test_project, budget={"caps": {"max_tokens": "1000"}})

    async def check(scope, context=None):
        task = await _scoped_task(db_session, test_project, run, scope, context)
        return await effective_authorization(db_session, goal, run, task)

    undeclared = await check(None)
    assert (undeclared.allowed, undeclared.workspace_mutation) == (True, "direct")
    assert undeclared.model_spend_remaining == {"max_tokens": Decimal("1000")}
    assert _denial(await check({})) == (False, "model_spend_not_granted", "revise_task_scope")
    assert _denial(await check({"model_spend": "yes"})) == (False, "model_spend_not_granted", "revise_task_scope")
    spend = {"model_spend": {"allowed": True, "remaining_by_dimension": {"max_tokens": "10", "max_turns": "2"}}}
    narrowed = await check({**spend, "tool_constraints": ["read_docs_only"]})
    assert (narrowed.allowed, narrowed.workspace_mutation) == (True, "read_only")
    assert narrowed.model_spend_remaining == {"max_tokens": Decimal("10"), "max_turns": Decimal("2")}
    assert narrowed.tool_constraints == ["no_external_writes", "no_live_sends", "read_docs_only"]
    assert _denial(await check({**spend, "external_writes": {"allowed": True, "targets": ["github"]}})) == (
        False, "external_write_not_granted", "ask_human")
    assert _denial(await check({**spend, "live_sends": {"allowed": True, "channels": ["slack"]}})) == (
        False, "live_send_not_granted", "ask_human")
    assert (await check({**spend, "workspace_mutation": "direct"})).allowed

    staged = {"roadmap": {"mutates_shared_state": True, "no_publish_before_integration": True,
                          "staging_boundary": {"type": "git_worktree", "identifier": "w", "reversible": True}}}
    over = await check({**spend, "workspace_mutation": "direct"}, staged)
    assert _denial(over) == (False, "workspace_mutation_exceeds_policy", "ask_human")
    within = await check({**spend, "workspace_mutation": "staged"}, staged)
    assert (within.allowed, within.workspace_mutation) == (True, "staged")
    assert "no_publish_before_integration" in within.tool_constraints


@pytest.mark.asyncio
async def test_effective_authorization_declared_zero_allowance_denied(db_session, test_project):
    from huddleroom.services.orchestration_authority_service import effective_authorization

    goal, run = await _started_root(db_session, test_project, budget={"caps": {"max_tokens": "1000"}})
    zero = {"model_spend": {"allowed": True, "remaining_by_dimension": {"max_tokens": "0"}}}
    task = await _scoped_task(db_session, test_project, run, zero)
    assert _denial(await effective_authorization(db_session, goal, run, task)) == (
        False, "model_spend_not_granted", "revise_task_scope")


@pytest.mark.asyncio
async def test_effective_authorization_baseline_rank_and_team(db_session, test_project, test_agent):
    from tests.test_orchestration_goal_shape import _goal_run
    from huddleroom.services.orchestration_authority_service import effective_authorization

    goal, run = await _goal_run(db_session, test_project, started=False)
    spend = {"model_spend": {"allowed": True}}
    staged = await _scoped_task(db_session, test_project, run, {**spend, "workspace_mutation": "staged"})
    assert _denial(await effective_authorization(db_session, goal, run, staged)) == (
        False, "workspace_mutation_exceeds_policy", "ask_human")

    started, started_run = await _started_root(db_session, test_project)
    context = {"roadmap": {"mutates_shared_state": False}, "team": {"agent_ids": [str(uuid.uuid4())]}}
    outsider = await _scoped_task(db_session, test_project, started_run, None, context, assigned_to=test_agent.id)
    assert _denial(await effective_authorization(db_session, started, started_run, outsider)) == (
        False, "agent_outside_accepted_team", "reassign_task")
    context["team"]["agent_ids"].append(str(test_agent.id))
    member = await _scoped_task(db_session, test_project, started_run, None, context, assigned_to=test_agent.id)
    assert (await effective_authorization(db_session, started, started_run, member)).allowed
    # Outcome work without a Roadmap context gets no new team restriction.
    plain = await _scoped_task(db_session, test_project, started_run, None, {"team": {"agent_ids": []}},
                               assigned_to=test_agent.id)
    assert (await effective_authorization(db_session, started, started_run, plain)).allowed


@pytest.mark.parametrize("gate_type,required,source_type,verdict,terminal,expected", [
    ("work_completed", {}, "task", "accepted", None, ("delivery", "final", "accepted")),
    ("work_completed", {}, "verification", "rejected", None, ("verification", "final", "rejected")),
    ("plan_accepted", {}, "artifact", "candidate", None, ("readiness", "final", "pending")),
    ("roadmap_plan_approval", None, "artifact", "candidate", "reclassified", ("readiness", "final", "superseded")),
    ("roadmap_integration", {}, "child_goal", "accepted", None, ("integration", "final", "accepted")),
    ("adhoc_work_accepted", {"evidence_role": "discovery", "review_stage": "preliminary"}, "task", "accepted", None,
     ("discovery", "preliminary", "accepted")),
    ("roadmap_integration", {"evidence_role": "bogus", "review_stage": "draft"}, "task", "candidate", "cancelled",
     ("integration", "final", "pending")),
])
def test_evidence_semantics_table(gate_type, required, source_type, verdict, terminal, expected):
    from huddleroom.services.orchestration_dependencies import evidence_semantics

    assert evidence_semantics(gate_type, required, source_type, verdict, terminal) == dict(
        zip(("evidence_role", "review_stage", "acceptance_verdict"), expected))


def test_plan_gate_type_is_a_readiness_gate():
    from huddleroom.services.orchestration_dependencies import READINESS_GATE_TYPES
    from huddleroom.services.orchestration_service import PLAN_GATE_TYPE

    assert PLAN_GATE_TYPE in READINESS_GATE_TYPES


async def _actions(db, run, action_type):
    return (await db.scalars(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == action_type))).all()


@pytest.mark.asyncio
async def test_restart_records_one_action_per_fingerprint_and_authorizes(db_session, test_project):
    from huddleroom.services.orchestration_authority_service import effective_authorization

    svc = OrchestrationService()
    goal, run = await _seed_baseline_terminal_run(db_session, test_project)
    await svc.tick(db_session, run.id)
    await svc.start_run(db_session, test_project.id, goal.id, actor="human:a")
    frozen = run.goal_contract_fingerprint
    (start,) = await _actions(db_session, run, "authorize_execution")
    assert start.request["contract_fingerprint"] == frozen
    assert (start.request["goal_type"], start.request["planning_mode"]) == (goal.goal_type, goal.planning_mode)

    await svc.start_run(db_session, test_project.id, goal.id, actor="human:a")  # unchanged contract
    assert await _actions(db_session, run, "reauthorize_execution") == []

    goal.objective = "Changed after Start"
    assert (await effective_authorization(db_session, goal, run)).reason_code == "contract_changed_since_start"
    await svc.start_run(db_session, test_project.id, goal.id, actor="human:b")
    await svc.start_run(db_session, test_project.id, goal.id, actor="human:c")  # same fingerprint
    (re1,) = await _actions(db_session, run, "reauthorize_execution")
    fp1 = compute_goal_contract_fingerprint(goal)
    assert re1.status == "completed"
    assert re1.request == {"actor": "human:b", "contract_fingerprint": fp1}
    assert (await effective_authorization(db_session, goal, run)).allowed

    goal.objective = "Changed again"
    await svc.start_run(db_session, test_project.id, goal.id, actor="human:b")
    assert len(await _actions(db_session, run, "reauthorize_execution")) == 2
    assert len(await _actions(db_session, run, "authorize_execution")) == 1
    assert run.goal_contract_fingerprint == frozen  # run column is never rewritten


@pytest.mark.asyncio
async def test_authorized_tick_freezes_legacy_null_fingerprint_on_roots_only(db_session, test_project, owned_session_factory):
    adopt_session(db_session)
    svc = OrchestrationService()
    goal, run = await _seed_baseline_terminal_run(db_session, test_project)
    await svc.tick(db_session, run.id)
    run.phase = "authorized"  # legacy Start: authorized without a frozen fingerprint
    assert run.goal_contract_fingerprint is None
    await svc.tick(db_session, run.id)
    await db_session.refresh(run)
    assert run.goal_contract_fingerprint == compute_goal_contract_fingerprint(goal)

    _, _, _, child, child_run = await _released(db_session, test_project)
    assert child.parent_goal_id is not None
    await db_session.execute(update(OrchestrationRun).where(OrchestrationRun.id == child_run.id).values(
        phase="authorized", goal_contract_fingerprint=None))  # column validator forbids ORM writes
    await db_session.refresh(child_run)
    assert child_run.goal_contract_fingerprint is None
    await svc.tick(db_session, child_run.id)
    await db_session.refresh(child_run)
    assert child_run.goal_contract_fingerprint is None


# --- activation hook (activate_ready is_blocked) ---

async def _started_with_backlog_task(db, project, *, owns_setup=False):
    from huddleroom.models.task import Task
    from tests.test_orchestration_runtime_e2e import _agent

    svc = OrchestrationService()
    goal, run = await _seed_baseline_terminal_run(db, project)
    if owns_setup:
        await db.commit()
    await svc.tick(db, run.id)
    await svc.start_run(db, project.id, goal.id, actor="human:a")
    agent = _agent("activation", ["implementation"])
    db.add(agent)
    await db.flush()
    task = Task(project_id=project.id, title="t", status="backlog", assigned_to=agent.id,
                metadata_={"orchestration": {"run_id": str(run.id)}})
    db.add(task)
    await db.flush()
    return svc, goal, run, task


async def _session_count(db, task):
    from sqlalchemy import func

    from huddleroom.models.session import Session

    return await db.scalar(select(func.count()).select_from(Session).where(Session.task_id == task.id))


@pytest.mark.asyncio
async def test_stale_start_keeps_task_queued_with_authorization_wait_until_restart(db_session, test_project):
    svc, goal, run, task = await _started_with_backlog_task(db_session, test_project)
    goal.objective = "Changed after Start"
    scope = f"authorization:{task.id}"
    run.active_blockers = [{"kind": "authorization", "scope": scope, "reason_code": "contract_changed_since_start"}]
    await db_session.flush()

    assert await svc._start_released_tasks(db_session, goal, run) == 0
    assert task.status == "backlog" and await _session_count(db_session, task) == 0
    assert task.metadata_["orchestration"]["authorization_wait"] == {
        "reason_code": "contract_changed_since_start", "required_decision": "start", "authority_source": "root_start",
    }
    assert task.metadata_["orchestration"]["run_id"] == str(run.id)
    assert await svc._start_released_tasks(db_session, goal, run) == 0  # unchanged denial is stable
    assert run.active_blockers  # only an allowed activation clears the claim blocker

    await svc.start_run(db_session, test_project.id, goal.id, actor="human:a")  # re-Start
    assert await svc._start_released_tasks(db_session, goal, run) == 1
    assert task.status == "in_progress" and await _session_count(db_session, task) == 1
    assert "authorization_wait" not in task.metadata_["orchestration"]
    assert not [b for b in run.active_blockers if b.get("scope") == scope]


@pytest.mark.asyncio
async def test_authorization_is_checked_after_stop_and_dependency_checks(db_session, test_project):
    svc, goal, run, task = await _started_with_backlog_task(db_session, test_project)
    goal.objective = "Changed after Start"
    run.active_blockers = [{"kind": "repeated_failure", "reason": "stopped"}]
    await db_session.flush()
    assert await svc._activation_blocker(db_session, run, task) == "repeated_failure"
    assert "authorization_wait" not in task.metadata_["orchestration"]  # stop semantics unchanged, authority not probed


@pytest.mark.asyncio
async def test_corrupt_lineage_is_an_authorization_denial_not_a_tick_crash(db_session, test_project, monkeypatch):
    from fastapi import HTTPException

    from huddleroom.services.session_service import SessionService

    async def corrupt(self, db, task):  # a corrupt roadmap row cannot be built under the FKs
        raise HTTPException(status_code=409, detail="Roadmap task lineage is invalid")

    svc, goal, run, task = await _started_with_backlog_task(db_session, test_project)
    monkeypatch.setattr(SessionService, "_immutable_roadmap_context", corrupt)

    assert await svc._start_released_tasks(db_session, goal, run) == 0
    assert task.status == "backlog" and await _session_count(db_session, task) == 0
    assert task.metadata_["orchestration"]["authorization_wait"] == {
        "reason_code": "lineage_invalid", "required_decision": "ask_human", "authority_source": "root_start",
        "detail": "Roadmap task lineage is invalid",
    }


# --- 15f: fingerprint covers only the human-authored contract ---

@pytest.mark.asyncio
async def test_manager_selection_rewrites_do_not_change_contract(db_session, test_project):
    from tests.test_orchestration_runtime_e2e import _agent
    from huddleroom.services.orchestration_authority_service import effective_authorization

    goal, run = await _started_root(db_session, test_project)
    agent = _agent("manager", ["management"])
    db_session.add(agent)
    await db_session.flush()
    goal.manager_agent_id = agent.id          # what a post-Start manager-selection rerun writes
    goal.authority_model = "agent_manager"
    assert (await effective_authorization(db_session, goal, run)).allowed

    goal.success_criteria = [{"id": "c1", "text": "new criterion"}]
    result = await effective_authorization(db_session, goal, run)
    assert (result.allowed, result.reason_code) == (False, "contract_changed_since_start")


@pytest.mark.asyncio
async def test_post_start_manager_rewrite_does_not_block_activation(db_session, test_project):
    svc, goal, run, task = await _started_with_backlog_task(db_session, test_project)
    goal.manager_user_id = None
    goal.authority_model = "agent_manager"
    await db_session.flush()
    assert not await svc._activation_blocker(db_session, run, task)
    assert "authorization_wait" not in task.metadata_["orchestration"]
