import uuid

import pytest
from sqlalchemy.exc import IntegrityError

from huddleroom.models.orchestration import (
    OrchestrationGoal,
    OrchestrationRun,
    compute_goal_contract_fingerprint,
)


def _goal(project, **overrides):
    values = dict(
        project_id=project.id, objective="Ship X", original_request="Ship X",
        success_criteria=[], constraints={}, budget={},
    )
    values.update(overrides)
    return OrchestrationGoal(**values)


async def _persist(db, *rows):
    db.add_all(rows)
    await db.flush()


@pytest.mark.asyncio
@pytest.mark.parametrize("goal_type,mode", [
    ("outcome", "implementation"), ("roadmap", "roadmap"), ("continuous", "continuous_policy"),
])
async def test_orm_defaults_map_goal_type_to_legacy_shape(db_session, test_project, goal_type, mode):
    goal = _goal(test_project, goal_type=goal_type)
    await _persist(db_session, goal)

    assert (goal.goal_type_source, goal.planning_mode) == ("legacy", mode)
    assert goal.goal_type_rationale.strip() and goal.planning_rationale.strip()


@pytest.mark.asyncio
@pytest.mark.parametrize("overrides", [
    {"goal_type_source": "bogus"},
    {"planning_mode": "bogus"},
    {"goal_type_rationale": "   "},
    {"planning_rationale": ""},
    {"goal_type": "roadmap", "planning_mode": "implementation"},
    {"goal_type": "continuous", "planning_mode": "delivery"},
    {"goal_type": "outcome", "planning_mode": "roadmap"},
])
async def test_goal_shape_checks_reject_invalid_rows(db_session, test_project, overrides):
    db_session.add(_goal(test_project, **overrides))
    with pytest.raises(IntegrityError):
        await db_session.flush()


@pytest.mark.asyncio
async def test_outcome_accepts_delivery_and_implementation(db_session, test_project):
    goals = [_goal(test_project, planning_mode=mode) for mode in ("delivery", "implementation")]
    await _persist(db_session, *goals)


@pytest.mark.asyncio
async def test_run_rejects_self_supersession(db_session, test_project):
    goal = _goal(test_project)
    await _persist(db_session, goal)
    run_id = uuid.uuid4()
    db_session.add(OrchestrationRun(id=run_id, goal_id=goal.id, budget_state={}, supersedes_run_id=run_id))
    with pytest.raises(IntegrityError):
        await db_session.flush()


@pytest.mark.asyncio
async def test_run_supersedes_run_id_is_unique(db_session, test_project):
    goal = _goal(test_project)
    await _persist(db_session, goal)
    old = OrchestrationRun(goal_id=goal.id, budget_state={}, status="cancelled")
    await _persist(db_session, old)
    await _persist(db_session, OrchestrationRun(goal_id=goal.id, budget_state={}, status="cancelled", supersedes_run_id=old.id))
    db_session.add(OrchestrationRun(goal_id=goal.id, budget_state={}, status="cancelled", supersedes_run_id=old.id))
    with pytest.raises(IntegrityError):
        await db_session.flush()


@pytest.mark.asyncio
@pytest.mark.parametrize("overrides", [
    {"terminal_reason": "bogus", "status": "cancelled"},
    {"terminal_reason": "reclassified", "status": "running"},
    {"goal_type_snapshot": "bogus"},
    {"planning_mode_snapshot": "bogus"},
])
async def test_run_checks_reject_invalid_rows(db_session, test_project, overrides):
    goal = _goal(test_project)
    await _persist(db_session, goal)
    db_session.add(OrchestrationRun(goal_id=goal.id, budget_state={}, **overrides))
    with pytest.raises(IntegrityError):
        await db_session.flush()


@pytest.mark.asyncio
async def test_persisted_run_snapshots_are_frozen_but_null_can_be_set_once(db_session, test_project):
    goal = _goal(test_project)
    await _persist(db_session, goal)
    old = OrchestrationRun(goal_id=goal.id, budget_state={}, status="cancelled")
    other = OrchestrationRun(goal_id=goal.id, budget_state={}, status="cancelled")
    run = OrchestrationRun(goal_id=goal.id, budget_state={})
    await _persist(db_session, old, other, run)

    run.goal_type_snapshot = "outcome"  # NULL -> value
    run.goal_type_snapshot = "outcome"  # same value is a no-op
    run.planning_mode_snapshot = "implementation"
    run.goal_contract_fingerprint = "a" * 64
    run.supersedes_run_id = old.id
    await db_session.flush()

    for key, value in (
        ("goal_type_snapshot", "roadmap"), ("planning_mode_snapshot", "delivery"),
        ("goal_contract_fingerprint", "b" * 64), ("supersedes_run_id", other.id),
    ):
        with pytest.raises(ValueError, match=f"{key} is immutable"):
            setattr(run, key, value)


@pytest.mark.asyncio
async def test_goal_contract_fingerprint_tracks_contract_not_status_or_context(db_session, test_project):
    goal = _goal(test_project)
    await _persist(db_session, goal)
    base = compute_goal_contract_fingerprint(goal)
    assert len(base) == 64 and base == compute_goal_contract_fingerprint(goal)

    goal.status = "paused"
    goal.orchestrator_context = {"anything": 1}
    assert compute_goal_contract_fingerprint(goal) == base

    goal.objective = "Ship Y"
    changed = compute_goal_contract_fingerprint(goal)
    assert changed != base
    goal.planning_mode = "delivery"
    with_mode = compute_goal_contract_fingerprint(goal)
    assert with_mode not in (base, changed)
    goal.goal_type, goal.planning_mode = "continuous", "continuous_policy"
    assert compute_goal_contract_fingerprint(goal) not in (base, changed, with_mode)


@pytest.mark.asyncio
async def test_frozen_snapshot_validator_does_not_lazy_load_expired_instance(db_session, test_project):
    goal = _goal(test_project)
    run = OrchestrationRun(goal_id=goal.id, budget_state={})
    await _persist(db_session, goal)
    run.goal_id = goal.id
    await _persist(db_session, run)
    db_session.expire(run)

    # A lazy load here would raise MissingGreenlet; an unloaded value is treated as unknown.
    run.goal_type_snapshot = "outcome"
