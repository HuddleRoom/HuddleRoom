import pytest
from sqlalchemy import select

from huddleroom.models.orchestration import (
    GOAL_TYPE_VALUES,
    RUN_PHASE_VALUES,
    OrchestrationAction,
    OrchestrationGoal,
    OrchestrationRun,
)
from huddleroom.services.orchestration_process_service import OrchestrationProcessService
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.orchestration_warning_service import OrchestrationWarningService


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

    r2 = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/start",
        headers=auth_headers,
    )
    assert r2.status_code == 200                      # idempotent, not 409
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
