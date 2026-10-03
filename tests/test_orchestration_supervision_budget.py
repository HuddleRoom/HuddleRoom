from contextlib import asynccontextmanager
from types import SimpleNamespace
import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from huddleroom.models.orchestration import (
    OrchestrationAction,
    OrchestrationBudgetReservation,
    OrchestrationGoal,
    OrchestrationRun,
)
from huddleroom.models.session import Session
from huddleroom.models.task import Task

from huddleroom.services.orchestration_budget_service import OrchestrationBudgetService


def _goal(caps):
    return SimpleNamespace(budget={"caps": caps})


def test_snapshot_canonicalizes_finite_nonnegative_budget_caps():
    snapshot = OrchestrationBudgetService().snapshot(_goal({"max_tokens": "2.00", "max_turns": 3}))

    assert snapshot["caps"] == {"max_tokens": "2", "max_turns": "3"}
    assert snapshot["dimensions"] == ("max_tokens", "max_turns")


def test_reserve_commit_settle_is_idempotent_and_preserves_capacity():
    service = OrchestrationBudgetService()
    snapshot = service.snapshot(_goal({"max_tokens": 5}))
    action = SimpleNamespace(budget_ledger={})

    reserved = service.reserve(snapshot, action, {"max_tokens": 3})
    assert reserved["remaining"] == {"max_tokens": "2"}
    assert service.reserve(snapshot, action, {"max_tokens": 3}) == reserved
    assert service.commit(snapshot, action, {"max_tokens": 3})["committed"] == {"max_tokens": "3"}
    assert service.settle(snapshot, action, {"max_tokens": 2})["consumed"] == {"max_tokens": "2"}
    assert service.settle(snapshot, action, {"max_tokens": 2})["remaining"] == {"max_tokens": "3"}


def test_budget_action_replay_with_different_amount_is_rejected():
    service = OrchestrationBudgetService()
    snapshot = service.snapshot(_goal({"max_tokens": 5}))
    action = SimpleNamespace(budget_ledger={})
    service.reserve(snapshot, action, {"max_tokens": 3})

    with pytest.raises(HTTPException, match="conflicts"):
        service.reserve(snapshot, action, {"max_tokens": 4})


def test_settlement_cannot_replace_a_final_observation_with_another_observation():
    """A retry may replay its observation, but cannot charge a second final result."""
    service = OrchestrationBudgetService()
    snapshot = service.snapshot(_goal({"max_tokens": 5}))
    action = SimpleNamespace(budget_ledger={})
    service.reserve(snapshot, action, {"max_tokens": 3})
    service.commit(snapshot, action)
    service.settle(snapshot, action, {"max_tokens": 2}, observation_id="session:one")

    with pytest.raises(HTTPException, match="conflicts"):
        service.settle(snapshot, action, {"max_tokens": 1}, observation_id="session:two")


def test_commit_replay_after_final_settlement_is_unchanged():
    service = OrchestrationBudgetService()
    snapshot = service.snapshot(_goal({"max_tokens": 5}))
    action = SimpleNamespace(budget_ledger={})
    service.reserve(snapshot, action, {"max_tokens": 3})
    service.commit(snapshot, action)
    settled = service.settle(snapshot, action, {"max_tokens": 2}, observation_id="session:one")

    assert service.commit(snapshot, action) == settled
    assert snapshot["committed"] == {"max_tokens": "0"}
    assert snapshot["consumed"] == {"max_tokens": "2"}


def test_settlement_releases_an_uncommitted_reservation_without_negative_commitment():
    service = OrchestrationBudgetService()
    snapshot = service.snapshot(_goal({"max_tokens": 5}))
    action = SimpleNamespace(budget_ledger={})
    service.reserve(snapshot, action, {"max_tokens": 3})

    service.settle(snapshot, action, {"max_tokens": 2})

    assert snapshot["reserved"] == {"max_tokens": "0"}
    assert snapshot["committed"] == {"max_tokens": "0"}
    assert snapshot["remaining"] == {"max_tokens": "3"}


def test_zero_final_settlement_is_terminal_and_exactly_replayable():
    service = OrchestrationBudgetService()
    snapshot = service.snapshot(_goal({"max_tokens": 5}))
    action = SimpleNamespace(budget_ledger={})
    service.reserve(snapshot, action, {"max_tokens": 3})
    service.commit(snapshot, action)
    settled = service.settle(snapshot, action, {"max_tokens": 0}, observation_id="session:one")

    assert service.commit(snapshot, action) == settled
    assert service.settle(snapshot, action, {"max_tokens": 0}, observation_id="session:one") == settled
    with pytest.raises(HTTPException, match="conflicts"):
        service.settle(snapshot, action, {"max_tokens": 1}, observation_id="session:two")


def test_discretionary_reservation_keeps_the_closeout_allowance_inside_cap():
    service = OrchestrationBudgetService()
    snapshot = service.snapshot(_goal({"max_tokens": 3}))

    service.reserve(snapshot, SimpleNamespace(budget_ledger={}), {"max_tokens": 2})
    with pytest.raises(HTTPException, match="remaining"):
        service.reserve(snapshot, SimpleNamespace(budget_ledger={}), {"max_tokens": 1})


def test_unknown_measurement_fails_closed_for_discretionary_reserve():
    service = OrchestrationBudgetService()
    snapshot = service.snapshot(_goal({"max_tokens": 5}))
    snapshot["measurement_complete"] = False

    with pytest.raises(HTTPException, match="measurement"):
        service.reserve(snapshot, SimpleNamespace(budget_ledger={}), {"max_tokens": 1})


def test_closeout_reserve_stays_inside_the_same_capacity_pool():
    service = OrchestrationBudgetService()
    snapshot = service.snapshot(_goal({"max_tokens": 3}))
    service.reserve(snapshot, SimpleNamespace(budget_ledger={}), {"max_tokens": 2})

    with pytest.raises(HTTPException, match="remaining"):
        service.reserve(snapshot, SimpleNamespace(budget_ledger={}), {"max_tokens": 2}, closeout=True)


def test_settlement_overage_marks_run_budget_state_exceeded():
    service = OrchestrationBudgetService()
    snapshot = service.snapshot(_goal({"max_tokens": 2}))
    action = SimpleNamespace(budget_ledger={})
    run = SimpleNamespace(budget_state={})
    service.reserve(snapshot, action, {"max_tokens": 2}, closeout=True)
    service.commit(snapshot, action)
    service.settle(snapshot, action, {"max_tokens": 3})

    service.apply_overage(run, snapshot)
    assert run.budget_state["status"] == "exceeded"


@pytest.mark.asyncio
async def test_persisted_snapshot_merges_run_budget_state(db_session, test_project):
    goal = OrchestrationGoal(
        project_id=test_project.id, objective="budget", original_request="budget",
        budget={"caps": {"max_tokens": 2}},
    )
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id, status="running", phase="authorized")
    db_session.add(run)
    await db_session.flush()

    snapshot = await OrchestrationBudgetService().snapshot_for_run(db_session, goal, run)

    assert snapshot["remaining"] == {"max_tokens": "2"}
    assert run.budget_state["consumed"] == {"max_tokens": "0"}


@pytest.mark.asyncio
async def test_action_ledger_uses_disjoint_amount_maps_and_final_measurement_identity(
    db_session, test_project,
):
    goal = OrchestrationGoal(
        project_id=test_project.id, objective="budget", original_request="budget",
        budget={"caps": {"max_tokens": 5}},
    )
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id, status="running", phase="authorized")
    db_session.add(run)
    await db_session.flush()
    action = OrchestrationAction(
        run_id=run.id, idempotency_key="ledger-shape", action_type="request_plan", request={},
    )
    db_session.add(action)
    await db_session.flush()

    service = OrchestrationBudgetService()
    await service.reserve_action_budget(db_session, goal, run, action, {"max_tokens": 3}, enforceable=True)
    await service.commit_action_budget(db_session, goal, run, action)
    await service.settle_action_budget(
        db_session, goal, run, action, {"max_tokens": 2},
        measurement_complete=True, observation_id="session:test:usage-final",
    )

    assert action.budget_ledger == {
        "allocation": {"max_tokens": "3"},
        "reserved": {}, "committed": {}, "consumed": {"max_tokens": "2"},
        "usage_state": "known", "enforceability": "enforceable",
        "final_observation": "session:test:usage-final:final",
        "observed_measurements": ["session:test:usage-final:final"],
        "measurement_amounts": {"session:test:usage-final:final": {"max_tokens": "2"}},
    }


@pytest.mark.asyncio
async def test_parent_snapshot_counts_active_and_settled_discovery_reservations(db_session, test_project):
    goal = OrchestrationGoal(
        project_id=test_project.id, objective="budget", original_request="budget", goal_type="roadmap",
        budget={"caps": {"max_tokens": 10}},
    )
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id, status="running", phase="authorized")
    settled_run = OrchestrationRun(goal_id=goal.id, status="completed", phase="completed", cycle_key="settled")
    db_session.add_all([run, settled_run])
    await db_session.flush()
    db_session.add_all([
        OrchestrationBudgetReservation(
            parent_goal_id=goal.id, discovery_run_id=run.id, allocation={"max_tokens": "3"},
        ),
        OrchestrationBudgetReservation(
            parent_goal_id=goal.id, discovery_run_id=settled_run.id, allocation={"max_tokens": "4"},
            settled_spend={"max_tokens": "2"}, status="settled", settlement_reason="completed",
            settled_at=settled_run.created_at,
        ),
    ])
    await db_session.flush()

    snapshot = await OrchestrationBudgetService().snapshot_for_run(db_session, goal, run)

    assert snapshot["reserved"] == {"max_tokens": "3"}
    assert snapshot["consumed"] == {"max_tokens": "2"}
    assert snapshot["remaining"] == {"max_tokens": "5"}


async def _run_action_with_session(db_session, test_project, test_agent, *, ledger, session_status, metadata, cap=500):
    goal = OrchestrationGoal(
        project_id=test_project.id, objective="budget", original_request="budget",
        budget={"caps": {"max_tokens": cap}},
    )
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id, status="running", phase="authorized")
    task = Task(project_id=test_project.id, title="budgeted task", status="in_progress")
    db_session.add_all([run, task])
    await db_session.flush()
    action = OrchestrationAction(
        run_id=run.id, idempotency_key="budget-action", action_type="create_delegation_task", request={},
        status="completed", target_type="task", target_id=task.id, budget_ledger=ledger,
    )
    session = Session(
        task_id=task.id, agent_id=test_agent.id, project_id=test_project.id, adapter_type="test",
        status=session_status, metadata_=metadata,
    )
    db_session.add_all([action, session])
    await db_session.flush()
    if ledger:
        session.metadata_ = {
            **(session.metadata_ or {}), "orchestration": {"action_id": str(action.id)},
        }
        await db_session.flush()
    return goal, run, action


@pytest.mark.asyncio
async def test_snapshot_uses_committed_action_ledger_once_for_linked_active_session(
    db_session, test_project, test_agent,
):
    goal, run, _ = await _run_action_with_session(
        db_session, test_project, test_agent,
        ledger={"reserved": {}, "committed": {"max_tokens": "100"}, "consumed": {},
                "usage_state": "known", "enforceability": "enforceable"},
        session_status="running", metadata={"_run_config": {"max_tokens": 100, "timeout": 60}},
    )

    snapshot = await OrchestrationBudgetService().snapshot_for_run(db_session, goal, run)

    assert snapshot["committed"] == {"max_tokens": "100"}
    assert snapshot["remaining"] == {"max_tokens": "400"}


@pytest.mark.asyncio
async def test_active_commitment_exhaustion_does_not_mark_run_budget_exceeded(
    db_session, test_project, test_agent,
):
    goal, run, _ = await _run_action_with_session(
        db_session, test_project, test_agent,
        ledger={},
        session_status="running", metadata={"_run_config": {"max_tokens": 500, "timeout": 60}},
    )

    from huddleroom.services.orchestration_service import OrchestrationService

    assert not await OrchestrationService()._sync_measured_budget_exhaustion(db_session, goal, run)
    assert run.budget_state["status"] == "active"


@pytest.mark.asyncio
async def test_negative_remaining_protected_allowance_dispatches_as_capacity_conflict(
    db_session, test_project, test_agent,
):
    goal, run, _ = await _run_action_with_session(
        db_session, test_project, test_agent,
        ledger={}, session_status="completed", metadata={"token_count_in": 2, "token_count_out": 0}, cap=1,
    )
    service = OrchestrationBudgetService()
    allocation = await service.protected_action_allocation(db_session, goal, run)
    action = OrchestrationAction(
        run_id=run.id, idempotency_key="exhausted-protected", action_type="request_verification", request={},
    )
    db_session.add(action)
    await db_session.flush()

    assert allocation == {"max_tokens": "0"}
    with pytest.raises(HTTPException) as exc:
        await service.reserve_action_budget(db_session, goal, run, action, allocation, closeout=True)
    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_snapshot_for_run_prefers_explicit_child_budget_caps_over_reservation(
    db_session, test_project,
):
    parent = OrchestrationGoal(
        project_id=test_project.id, objective="parent", original_request="parent", goal_type="continuous",
        budget={"caps": {"max_tokens": 100}},
    )
    db_session.add(parent)
    await db_session.flush()
    child = OrchestrationGoal(
        project_id=test_project.id, objective="child", original_request="child", parent_goal_id=parent.id,
        continuous_origin_key="child", parent_contract_snapshot={}, goal_delta={},
    )
    db_session.add(child)
    await db_session.flush()
    db_session.add(OrchestrationBudgetReservation(
        parent_goal_id=parent.id, child_goal_id=child.id, continuous_origin_key="child",
        allocation={"max_tokens": "80"},
    ))
    run = OrchestrationRun(
        goal_id=child.id, status="running", phase="authorized", budget_state={"caps": {"max_tokens": 10}},
    )
    db_session.add(run)
    await db_session.flush()

    snapshot = await OrchestrationBudgetService().snapshot_for_run(db_session, child, run)

    assert snapshot["caps"] == {"max_tokens": "10"}
    assert snapshot["remaining"] == {"max_tokens": "10"}


@pytest.mark.asyncio
async def test_snapshot_persists_settled_action_ledger_without_counting_linked_session_twice(
    db_session, test_project, test_agent,
):
    goal, run, _ = await _run_action_with_session(
        db_session, test_project, test_agent,
        ledger={"reserved": {}, "committed": {}, "consumed": {"max_tokens": "60"},
                "usage_state": "known", "enforceability": "enforceable"},
        session_status="completed", metadata={"token_count_in": 90, "token_count_out": 10},
    )
    service = OrchestrationBudgetService()

    first = await service.snapshot_for_run(db_session, goal, run)
    await db_session.flush()
    db_session.expunge_all()
    persisted_goal = await db_session.get(OrchestrationGoal, goal.id)
    persisted_run = await db_session.get(OrchestrationRun, run.id)
    restarted = await service.snapshot_for_run(db_session, persisted_goal, persisted_run)

    assert first["consumed"] == {"max_tokens": "60"}
    assert restarted["consumed"] == {"max_tokens": "60"}


@pytest.mark.asyncio
async def test_snapshot_rejects_another_unbound_session_on_the_same_ledger_task(db_session, test_project, test_agent):
    goal, run, action = await _run_action_with_session(
        db_session, test_project, test_agent,
        ledger={"reserved": {}, "committed": {}, "consumed": {"max_tokens": "60"},
                "usage_state": "known", "enforceability": "enforceable"},
        session_status="completed", metadata={"token_count_in": 30, "token_count_out": 30},
    )
    ledger_session = await db_session.scalar(select(Session).where(Session.task_id == action.target_id))
    ledger_session.metadata_ = {
        **ledger_session.metadata_, "orchestration": {"action_id": str(action.id)},
    }
    # The newer retry has its own action binding; only the ledger-bound session
    # is excluded from direct spend.
    task_id = action.target_id
    db_session.add(Session(
        task_id=task_id, agent_id=test_agent.id, project_id=test_project.id, adapter_type="test",
        status="completed", metadata_={"token_count_in": 10, "token_count_out": 10,
            "orchestration": {"action_id": str(uuid.uuid4())}},
    ))
    await db_session.flush()

    from huddleroom.services.orchestration_budget_service import BudgetMeasurementError

    with pytest.raises(BudgetMeasurementError):
        await OrchestrationBudgetService().snapshot_for_run(db_session, goal, run)


@pytest.mark.asyncio
async def test_settle_action_budget_after_pause_uses_goal_lock_but_control_gates_mutations(
    db_session, test_project, test_agent, monkeypatch,
):
    goal, run, action = await _run_action_with_session(
        db_session, test_project, test_agent,
        ledger={"reserved": {}, "committed": {"max_tokens": "100"}, "consumed": {},
                "usage_state": "known", "enforceability": "enforceable"},
        session_status="running", metadata={"_run_config": {"max_tokens": 100, "timeout": 60}},
    )
    locked = []

    @asynccontextmanager
    async def lock(_, __, goal_id):
        locked.append(goal_id)
        yield

    from huddleroom.services.orchestration_service import OrchestrationService

    monkeypatch.setattr(OrchestrationService, "_lock_goal_for_baseline_transition", lock)
    goal.status = run.status = "paused"

    service = OrchestrationBudgetService()
    settled = await service.settle_action_budget(
        db_session, goal, run, action, {"max_tokens": 60},
    )

    assert locked == [goal.id]
    assert settled["consumed"] == {"max_tokens": "60"}
    with pytest.raises(HTTPException, match="not budget-runnable"):
        await service.reserve_action_budget(db_session, goal, run, action, {"max_tokens": 1})
    with pytest.raises(HTTPException, match="not budget-runnable"):
        await service.commit_action_budget(db_session, goal, run, action)


@pytest.mark.parametrize("caps", [
    {"max_tokens": "NaN"}, {"max_tokens": "Infinity"}, {"max_tokens": -1},
])
def test_snapshot_rejects_nonfinite_or_negative_caps(caps):
    with pytest.raises(HTTPException, match="Invalid amount"):
        OrchestrationBudgetService().snapshot(_goal(caps))
