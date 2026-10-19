import uuid

from huddleroom.config import settings
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGoal, OrchestrationRun
from huddleroom.services.orchestration_roadmap_service import OrchestrationRoadmapService
from huddleroom.services.orchestration_service import OrchestrationService


async def _setup(db, project, plan_state=None):
    goal = OrchestrationGoal(project_id=project.id, objective="Roadmap", status="active", goal_type="roadmap")
    db.add(goal)
    await db.flush()
    run = OrchestrationRun(
        goal_id=goal.id, status="running", phase="authorized",
        plan_state={"status": "accepted"} if plan_state is None else plan_state,
    )
    db.add(run)
    await db.flush()
    # This isolated loop fixture models the unmarked pre-G3 authorization shape.
    db.add(OrchestrationAction(
        run_id=run.id, idempotency_key=f"run:{run.id}:kind:authorize_execution",
        action_type="authorize_execution", status="completed", request={},
    ))
    await db.flush()
    return goal, run


def _action(db, run, status="completed", action_type="request_verification"):
    action = OrchestrationAction(
        run_id=run.id, idempotency_key=f"k:{uuid.uuid4()}", action_type=action_type, status=status,
    )
    db.add(action)
    return action


def _wire(monkeypatch, db, run, results, *, cap=3):
    """results: list of (step, action_type_or_None, status) — one per _advance_once call."""
    orchestration = OrchestrationService()
    roadmap = OrchestrationRoadmapService(orchestration)
    monkeypatch.setattr(settings, "orchestration_max_actions_per_tick", cap)
    calls = {"n": 0}

    async def once(_db, _goal, _run):
        step, action_type, status = results[min(calls["n"], len(results) - 1)]
        calls["n"] += 1
        if action_type is None:
            return {"step": step, "reason": "needs_attention"}
        action = _action(_db, _run, status=status, action_type=action_type)
        await _db.flush()
        return {"step": step, "action_id": str(action.id)}

    monkeypatch.setattr(roadmap, "_advance_once", once)

    return roadmap, calls


async def test_roadmap_decision_steps_loop_until_noop_or_cap(db_session, test_project, monkeypatch):
    goal, run = await _setup(db_session, test_project)
    roadmap, calls = _wire(
        monkeypatch, db_session, run, [("replan_decision", "request_verification", "completed")] * 5, cap=3,
    )
    result = await roadmap.advance(db_session, goal, run)
    assert calls["n"] == 3
    assert len(result["action_ids"]) == 3
    assert result["action_id"] == result["action_ids"][0]
    assert result["step"] == "replan_decision"

    goal2, run2 = await _setup(db_session, test_project)
    roadmap2, calls2 = _wire(
        monkeypatch, db_session, run2,
        [("terminal_item_decision", "request_verification", "completed"), ("integration_decision", "noop", "completed")],
        cap=5,
    )
    result2 = await roadmap2.advance(db_session, goal2, run2)
    assert calls2["n"] == 2
    assert len(result2["action_ids"]) == 2


async def test_release_item_and_waiting_steps_never_loop(db_session, test_project, monkeypatch):
    goal, run = await _setup(db_session, test_project)
    roadmap, calls = _wire(monkeypatch, db_session, run, [("release_item", "request_verification", "completed")] * 3)
    result = await roadmap.advance(db_session, goal, run)
    assert calls["n"] == 1
    assert "action_ids" not in result

    goal2, run2 = await _setup(db_session, test_project)
    roadmap2, calls2 = _wire(monkeypatch, db_session, run2, [("waiting", None, None)] * 3)
    result2 = await roadmap2.advance(db_session, goal2, run2)
    assert calls2["n"] == 1
    assert result2 == {"step": "waiting", "reason": "needs_attention"}


async def test_inner_plan_decision_result_does_not_trigger_outer_loop(db_session, test_project, monkeypatch):
    goal, run = await _setup(db_session, test_project)
    orchestration = OrchestrationService()
    roadmap = OrchestrationRoadmapService(orchestration)
    calls = {"n": 0}
    action = _action(db_session, run)
    await db_session.flush()

    async def no_version(*_a, **_k):
        return None

    async def inner(_db, _goal, _run):
        calls["n"] += 1
        return {"step": "plan_decision", "action_id": str(action.id)}

    monkeypatch.setattr(roadmap, "current_version", no_version)
    monkeypatch.setattr(orchestration, "_advance_authorized_execution", inner)
    result = await roadmap.advance(db_session, goal, run)
    assert calls["n"] == 1
    assert result == {"step": "plan_decision", "action_id": str(action.id)}


async def test_single_roadmap_result_shape_unchanged(db_session, test_project, monkeypatch):
    goal, run = await _setup(db_session, test_project)
    roadmap, _ = _wire(monkeypatch, db_session, run, [("replan_decision", "noop", "completed")])
    result = await roadmap.advance(db_session, goal, run)
    assert set(result) == {"step", "action_id"}
    assert result["step"] == "replan_decision"


def _grant_stub(monkeypatch, scope):
    from huddleroom.services import orchestration_roadmap_service as roadmap_module
    from huddleroom.services.orchestration_authority_service import ExecutionGrant

    async def grant(_db, _goal, _run):
        return None if scope is None else ExecutionGrant(action=None, scope=scope)

    monkeypatch.setattr(roadmap_module, "execution_grant", grant)


async def test_planning_only_without_version_requests_plan(db_session, test_project, monkeypatch):
    goal, run = await _setup(db_session, test_project, plan_state={"status": "requested"})
    orchestration = OrchestrationService()
    roadmap = OrchestrationRoadmapService(orchestration)
    _grant_stub(monkeypatch, "planning_only")
    sentinel = {"step": "request_plan_sentinel"}
    calls = {"plan": 0, "renew": 0}

    async def no_version(*_a, **_k):
        return None

    async def plan(*_a, **_k):
        calls["plan"] += 1
        return sentinel

    async def renew(*_a, **_k):
        calls["renew"] += 1

    monkeypatch.setattr(roadmap, "current_version", no_version)
    monkeypatch.setattr(orchestration, "_advance_authorized_execution", plan)
    monkeypatch.setattr(roadmap, "renew_corrected_children", renew)
    assert await roadmap._advance_once(db_session, goal, run) == sentinel
    assert calls == {"plan": 1, "renew": 0}


async def test_planning_only_accepted_plan_without_version_waits_for_execution_scope(db_session, test_project, monkeypatch):
    goal, run = await _setup(db_session, test_project)
    orchestration = OrchestrationService()
    roadmap = OrchestrationRoadmapService(orchestration)
    _grant_stub(monkeypatch, "planning_only")
    calls = {"plan": 0, "renew": 0}

    async def no_version(*_a, **_k):
        return None

    async def plan(*_a, **_k):
        calls["plan"] += 1
        raise AssertionError("accepted plan under planning-only must not reach the outcome release path")

    async def renew(*_a, **_k):
        calls["renew"] += 1

    monkeypatch.setattr(roadmap, "current_version", no_version)
    monkeypatch.setattr(orchestration, "_advance_authorized_execution", plan)
    monkeypatch.setattr(roadmap, "renew_corrected_children", renew)
    assert await roadmap._advance_once(db_session, goal, run) == {"step": "waiting_execution_scope"}
    assert calls == {"plan": 0, "renew": 0}


async def test_execution_grant_without_version_advances_and_renews_once(db_session, test_project, monkeypatch):
    goal, run = await _setup(db_session, test_project)
    orchestration = OrchestrationService()
    roadmap = OrchestrationRoadmapService(orchestration)
    _grant_stub(monkeypatch, "execution")
    sentinel = {"step": "request_plan_sentinel"}
    calls = {"plan": 0, "renew": 0}

    async def no_version(*_a, **_k):
        return None

    async def plan(*_a, **_k):
        calls["plan"] += 1
        return sentinel

    async def renew(*_a, **_k):
        calls["renew"] += 1

    monkeypatch.setattr(roadmap, "current_version", no_version)
    monkeypatch.setattr(orchestration, "_advance_authorized_execution", plan)
    monkeypatch.setattr(roadmap, "renew_corrected_children", renew)
    assert await roadmap._advance_once(db_session, goal, run) == sentinel
    assert calls == {"plan": 1, "renew": 1}


async def test_planning_only_with_accepted_version_releases_nothing(db_session, test_project, monkeypatch):
    goal, run = await _setup(db_session, test_project)
    orchestration = OrchestrationService()
    roadmap = OrchestrationRoadmapService(orchestration)
    _grant_stub(monkeypatch, "planning_only")
    calls = {"plan": 0, "release": 0, "settle": 0}

    async def version(*_a, **_k):
        return object()

    async def plan(*_a, **_k):
        calls["plan"] += 1
        raise AssertionError("planning-only must not re-enter execution once a version exists")

    async def renew(*_a, **_k):
        calls["release"] += 1

    async def settle(*_a, **_k):
        calls["settle"] += 1
        return 0

    monkeypatch.setattr(roadmap, "current_version", version)
    monkeypatch.setattr(orchestration, "_advance_authorized_execution", plan)
    monkeypatch.setattr(roadmap, "renew_corrected_children", renew)
    monkeypatch.setattr(roadmap, "settle_finished_children", settle)
    assert await roadmap._advance_once(db_session, goal, run) == {"step": "waiting_execution_scope"}
    assert calls == {"plan": 0, "release": 0, "settle": 0}


async def test_missing_grant_waits_for_execution_scope(db_session, test_project, monkeypatch):
    goal, run = await _setup(db_session, test_project)
    orchestration = OrchestrationService()
    roadmap = OrchestrationRoadmapService(orchestration)
    _grant_stub(monkeypatch, None)
    calls = {"plan": 0}

    async def plan(*_a, **_k):
        calls["plan"] += 1
        raise AssertionError("no grant must not request a plan")

    monkeypatch.setattr(orchestration, "_advance_authorized_execution", plan)
    assert await roadmap._advance_once(db_session, goal, run) == {"step": "waiting_execution_scope"}
    assert calls["plan"] == 0
