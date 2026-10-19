import asyncio
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import uuid

import pytest
from tests.test_orchestration_recovery_persistence import owned_recovery_setup  # noqa: F401 -- fixture
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError

from huddleroom.models.orchestration import (
    OrchestrationAction, OrchestrationBudgetReservation, OrchestrationContinuousCandidate,
    OrchestrationGoal, OrchestrationRun, compute_goal_contract_fingerprint,
)
from huddleroom.models.event_log import EventLog
from huddleroom.models.project import Project
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.models.orchestration_process import OrchestrationProcessRun
from huddleroom.schemas.orchestration import OrchestrationContinuousPolicy, parse_discovery_batch
from huddleroom.schemas.session import SessionCreate
from huddleroom.services.orchestration_budget_service import BudgetMeasurementError, OrchestrationBudgetService
from huddleroom.services.orchestration_authority_service import reauthorize_execution_key
from huddleroom.services.orchestration_continuous_service import OrchestrationContinuousService
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.session_service import SessionClaimAttention, SessionService
from tests.test_orchestration_roster_mutation import sessions  # noqa: F401 -- fixture

# SQLAlchemy's dynamic function namespace is not statically callable to pylint.
# pylint: disable=not-callable


@pytest.mark.asyncio
async def test_stop_is_idempotent_and_returns_latest_drained_run(db_session, test_project):
    goal, run = await started_continuous(db_session, test_project)
    continuous = OrchestrationContinuousService(OrchestrationService())
    due = datetime.fromisoformat(goal.continuous_state["next_due_at"])
    await continuous.stop(
        db_session, test_project.id, goal.id, actor="human:test", reason="maintenance", now=due,
    )
    run.status = run.phase = "completed"
    run.completed_at = due
    stopped_goal, stopped_run = await continuous.stop(
        db_session, test_project.id, goal.id, actor="human:test", reason="maintenance", now=due,
    )
    assert stopped_goal.id == goal.id
    assert stopped_run.id == run.id
    assert stopped_goal.continuous_state["health"] == "stopped"


@pytest.mark.asyncio
async def test_successor_releases_pending_discovery_candidate_fifo(db_session, test_project, test_agent, monkeypatch):
    goal, cycle, task, session, continuous = await dispatched_discovery_source(
        db_session, test_project, test_agent, monkeypatch,
    )
    task.status = "done"
    session.status = "completed"
    session.output = '{"schema_version":1,"candidates":[{"origin_key":"source:1","objective":"Inspect","source_refs":["record:1"]}]}'
    session.metadata_ = {**session.metadata_, "token_count_in": 0, "token_count_out": 0,
                         "token_usage_complete": True, "_roadmap_elapsed_seconds": 0}

    original_reserve = OrchestrationBudgetService.reserve_continuous_child

    async def budget_wait(*_args, **_kwargs):
        raise HTTPException(status_code=409, detail="continuous_budget_wait")

    monkeypatch.setattr(OrchestrationBudgetService, "reserve_continuous_child", budget_wait)
    assert (await continuous.advance(db_session, goal, cycle))["outcome"] == "backlogged"
    pending = list(goal.continuous_state["pending_candidates"])
    successor = await OrchestrationService().get_active_run_for_goal(db_session, test_project.id, goal.id)
    monkeypatch.setattr(OrchestrationBudgetService, "reserve_continuous_child", original_reserve)

    assert await continuous.claim_due_cycle(
        db_session, goal.id, datetime.fromisoformat(goal.continuous_state["next_due_at"]),
    ) is successor
    assert goal.continuous_state["pending_candidates"] == []
    candidate = await db_session.scalar(select(OrchestrationContinuousCandidate).where(
        OrchestrationContinuousCandidate.parent_goal_id == goal.id,
    ))
    assert candidate.child_goal_id is not None
    assert pending[0]["candidate_id"] == str(candidate.id)
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
        OrchestrationAction.run_id == successor.id,
        OrchestrationAction.action_type == "release_continuous_child",
    )) == 1


@pytest.mark.asyncio
async def test_stop_clears_actionable_discovery_backlog_and_settles_source(db_session, test_project, test_agent, monkeypatch):
    goal, cycle, task, session, continuous = await dispatched_discovery_source(
        db_session, test_project, test_agent, monkeypatch,
    )
    goal.continuous_state = {**goal.continuous_state, "pending_candidates": [
        {"candidate_id": str(uuid.uuid4()), "reason": "capacity"},
    ]}
    await continuous.stop(db_session, test_project.id, goal.id, actor="human:test", reason="stop")

    reservation = await db_session.scalar(select(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.discovery_run_id == cycle.id,
    ))
    action = await db_session.scalar(select(OrchestrationAction).where(
        OrchestrationAction.run_id == cycle.id,
        OrchestrationAction.action_type == "stop_continuous",
    ))
    assert goal.continuous_state["pending_candidates"] == []
    assert action.request["pending_candidates"][0]["reason"] == "capacity"
    assert reservation.status == "settled" and reservation.settlement_reason == "cancelled"
    assert task.status == "cancelled" and session.status == "cancelled"


@pytest.mark.asyncio
async def test_pending_discovery_releases_before_new_source_assignment_validation(db_session, test_project, test_agent, monkeypatch):
    goal, cycle, task, session, continuous = await dispatched_discovery_source(
        db_session, test_project, test_agent, monkeypatch,
    )
    task.status = "done"
    session.status = "completed"
    session.output = '{"schema_version":1,"candidates":[{"origin_key":"source:1","objective":"Inspect","source_refs":["record:1"]}]}'
    session.metadata_ = {**session.metadata_, "token_count_in": 0, "token_count_out": 0,
                         "token_usage_complete": True, "_roadmap_elapsed_seconds": 0}

    async def budget_wait(*_args, **_kwargs):
        raise HTTPException(status_code=409, detail="continuous_budget_wait")

    monkeypatch.setattr(OrchestrationBudgetService, "reserve_continuous_child", budget_wait)
    await continuous.advance(db_session, goal, cycle)
    successor = await OrchestrationService().get_active_run_for_goal(db_session, test_project.id, goal.id)
    test_agent.is_active = False
    successor.active_blockers = [{"kind": "continuous_discovery_source_unrunnable"}]
    monkeypatch.undo()

    assert await continuous.claim_due_cycle(db_session, goal.id, datetime.fromisoformat(
        goal.continuous_state["next_due_at"],
    )) is None
    candidate = await db_session.scalar(select(OrchestrationContinuousCandidate).where(
        OrchestrationContinuousCandidate.parent_goal_id == goal.id,
    ))
    assert candidate.child_goal_id is not None


@pytest.mark.asyncio
async def test_legacy_direct_pending_uses_originating_frozen_policy(db_session, test_project, monkeypatch):
    goal, cycle = await started_continuous(db_session, test_project)
    continuous = OrchestrationContinuousService(OrchestrationService())
    due = datetime.fromisoformat(goal.continuous_state["next_due_at"])
    await continuous.claim_due_cycle(db_session, goal.id, due)

    async def budget_wait(*_args, **_kwargs):
        raise HTTPException(status_code=409, detail="continuous_budget_wait")

    original_reserve = OrchestrationBudgetService.reserve_continuous_child
    monkeypatch.setattr(OrchestrationBudgetService, "reserve_continuous_child", budget_wait)
    await continuous.advance(db_session, goal, cycle, due)
    monkeypatch.setattr(OrchestrationBudgetService, "reserve_continuous_child", original_reserve)
    goal.continuous_state["pending_candidates"][0].pop("policy")
    pending_version = goal.continuous_state["pending_candidates"][0]["policy_version"]
    successor = await OrchestrationService().get_active_run_for_goal(db_session, test_project.id, goal.id)
    await continuous.update_policy(db_session, test_project.id, goal.id,
        OrchestrationContinuousPolicy.model_validate(direct_policy(response_target_seconds=1)), actor="human:test")

    assert await continuous.release_pending_candidates(
        db_session, goal, successor, datetime.now(timezone.utc),
    ) is None
    child = await db_session.scalar(select(OrchestrationGoal).where(OrchestrationGoal.parent_goal_id == goal.id))
    assert child.orchestrator_context["continuous"]["policy_version"] == pending_version
    assert successor.id != cycle.id


@pytest.mark.asyncio
async def test_stop_missing_discovery_reservation_records_budget_integrity(db_session, test_project, test_agent, monkeypatch):
    goal, cycle, *_ = await dispatched_discovery_source(db_session, test_project, test_agent, monkeypatch)
    await db_session.execute(delete(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.discovery_run_id == cycle.id,
    ))
    await OrchestrationContinuousService(OrchestrationService()).stop(
        db_session, test_project.id, goal.id, actor="human:test", reason="stop",
    )
    assert any(item["kind"] == "budget_integrity" for item in cycle.active_blockers)


@pytest.mark.asyncio
async def test_pause_holds_terminal_discovery_output(db_session, test_project, test_agent, monkeypatch):
    goal, cycle, task, session, continuous = await dispatched_discovery_source(
        db_session, test_project, test_agent, monkeypatch,
    )
    task.status = "done"
    session.status = "completed"
    session.output = '{"schema_version":1,"candidates":[]}'
    await OrchestrationService().pause_goal(db_session, test_project.id, goal.id)
    assert (await continuous.advance(db_session, goal, cycle))["step"] == "control_held"


@pytest.mark.asyncio
async def test_direct_policy_update_clears_stale_discovery_source_blocker(db_session, test_project, test_agent):
    goal, run = await discovery_ready(db_session, test_project, test_agent)
    run.active_blockers = [{"kind": "continuous_discovery_source_unrunnable"}]

    await OrchestrationContinuousService(OrchestrationService()).update_policy(
        db_session, test_project.id, goal.id,
        OrchestrationContinuousPolicy.model_validate(direct_policy()), actor="human:test",
    )

    assert not any(item["kind"] == "continuous_discovery_source_unrunnable" for item in run.active_blockers)


@pytest.mark.asyncio
async def test_stop_direct_unclaimed_successor_does_not_create_budget_integrity(db_session, test_project):
    goal, run = await started_continuous(db_session, test_project)
    await OrchestrationContinuousService(OrchestrationService()).stop(
        db_session, test_project.id, goal.id, actor="human:test", reason="stop",
    )
    assert not any(item["kind"] == "budget_integrity" for item in run.active_blockers)


@pytest.mark.asyncio
async def test_pause_preserves_existing_caller_transaction(db_session, test_project, monkeypatch):
    goal, _ = await started_continuous(db_session, test_project)
    commits = []
    original_commit = db_session.commit

    async def record_commit():
        commits.append(True)
        await original_commit()

    monkeypatch.setattr(db_session, "commit", record_commit)
    original_root = db_session.sync_session.get_transaction()
    await OrchestrationService().pause_goal(db_session, test_project.id, goal.id)
    assert commits == []
    assert db_session.sync_session.get_transaction() is original_root
    assert original_root.is_active


@pytest.mark.asyncio
async def test_stop_after_cancel_rejects_without_reviving_goal(db_session, test_project):
    goal, run = await started_continuous(db_session, test_project)
    continuous = OrchestrationContinuousService(OrchestrationService())
    await OrchestrationService().cancel_goal(db_session, test_project.id, goal.id)
    state_before = deepcopy(goal.continuous_state)
    actions_before = await db_session.scalar(select(func.count()).select_from(OrchestrationAction))

    with pytest.raises(HTTPException) as exc:
        await continuous.stop(db_session, test_project.id, goal.id, actor="human:test", reason="too late")

    assert exc.value.status_code == 409
    assert goal.status == run.status == "cancelled"
    assert goal.continuous_state == state_before
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction)) == actions_before


@pytest.mark.asyncio
async def test_stopped_parent_settles_child_without_an_active_parent_run(db_session, test_project):
    goal, run = await started_continuous(db_session, test_project)
    continuous = OrchestrationContinuousService(OrchestrationService())
    due = datetime.fromisoformat(goal.continuous_state["next_due_at"])
    await continuous.claim_due_cycle(db_session, goal.id, due)
    await continuous.advance(db_session, goal, run, due)
    await continuous.stop(db_session, test_project.id, goal.id, actor="human:test", reason="drain", now=due)
    child = await db_session.scalar(select(OrchestrationGoal).where(OrchestrationGoal.parent_goal_id == goal.id))
    child_run = await db_session.scalar(select(OrchestrationRun).where(OrchestrationRun.goal_id == child.id))
    child.status = "completed"
    child_run.status = child_run.phase = "completed"
    run.status = run.phase = "completed"
    await continuous.claim_due_cycle(db_session, goal.id, due + timedelta(hours=1))
    reservation = await db_session.scalar(select(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.child_goal_id == child.id,
    ))
    assert reservation.status == "settled"

def direct_policy(**overrides):
    value = {
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
    value.update(overrides)
    return value


def discovery_policy(**overrides):
    value = direct_policy(
        adapter_filter=None,
        cycle_mode="discovery",
        discovery_source={
            "agent_id": str(uuid.uuid4()),
            "work_function": "research",
            "instructions": "List records",
            "source_filter": "Open records",
            "origin_key_rule": "record id",
            "access_requirements": ["read:records"],
            "max_candidates": 2,
            "timeout_seconds": 300,
            "budget": {
                "per_cycle": {"max_tokens": "100", "max_turns": "2", "max_hours": "0.5"},
                "rolling": {
                    "window_seconds": 3600,
                    "limits": {"max_tokens": "500", "max_turns": "10", "max_hours": "2"},
                },
            },
        },
    )
    value.update(overrides)
    return value


async def discovery_ready(db_session, test_project, test_agent, **overrides):
    goal, run = await continuous_ready(db_session, test_project)
    policy = discovery_policy(**overrides)
    policy["discovery_source"]["agent_id"] = str(test_agent.id)
    db_session.add(OrchestrationProcessRun(
        goal_id=goal.id, process_type="team_hierarchy", status="completed",
        trigger_reason="test",
        outputs={"role_to_agent": {"research": str(test_agent.id)}},
    ))
    await db_session.flush()
    await OrchestrationContinuousService(OrchestrationService()).update_policy(
        db_session, test_project.id, goal.id,
        OrchestrationContinuousPolicy.model_validate(policy), actor="human:test",
    )
    return goal, run


async def test_discovery_claim_refusal_keeps_due_slot_unmodified(db_session, test_project, test_agent):
    goal, run = await discovery_ready(db_session, test_project, test_agent)
    await OrchestrationService().start_run(db_session, test_project.id, goal.id, actor="human:test")
    due = goal.continuous_state["next_due_at"]
    test_agent.is_active = False

    assert await OrchestrationContinuousService(OrchestrationService()).claim_due_cycle(
        db_session, goal.id, datetime.fromisoformat(due),
    ) is None

    assert run.cycle_key is None
    assert goal.continuous_state["next_due_at"] == due
    assert not await db_session.scalar(select(OrchestrationAction.id).where(
        OrchestrationAction.run_id == run.id,
        OrchestrationAction.action_type == "claim_continuous_cycle",
    ))
    assert sum(item.get("kind") == "continuous_discovery_source_unrunnable" for item in run.active_blockers) == 1


async def test_discovery_start_translates_revoked_assignment_to_goal_not_runnable(
    db_session, test_project, test_agent,
):
    goal, _ = await discovery_ready(db_session, test_project, test_agent)
    test_agent.is_active = False

    with pytest.raises(HTTPException, match="goal_not_runnable"):
        await OrchestrationService().start_run(db_session, test_project.id, goal.id, actor="human:test")

    assert goal.continuous_state["started_at"] is None


async def test_discovery_full_backlog_refuses_without_claim_or_reservation(
    db_session, test_project, test_agent,
):
    goal, run = await discovery_ready(db_session, test_project, test_agent, max_backlog=1)
    await OrchestrationService().start_run(db_session, test_project.id, goal.id, actor="human:test")
    due = goal.continuous_state["next_due_at"]
    goal.continuous_state = {**goal.continuous_state, "pending_candidates": [{"origin_key": "held"}]}

    assert await OrchestrationContinuousService(OrchestrationService()).claim_due_cycle(
        db_session, goal.id, datetime.fromisoformat(due),
    ) is None

    assert run.cycle_key is None
    assert goal.continuous_state["next_due_at"] == due
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id,
        OrchestrationAction.action_type == "claim_continuous_cycle",
    )) == 0
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.discovery_run_id == run.id,
    )) == 0


async def test_revoked_discovery_assignment_settles_claimed_source_once(
    db_session, test_project, test_agent,
):
    goal, run = await discovery_ready(db_session, test_project, test_agent)
    service = OrchestrationService()
    await service.start_run(db_session, test_project.id, goal.id, actor="human:test")
    due = datetime.fromisoformat(goal.continuous_state["next_due_at"])
    continuous = OrchestrationContinuousService(service)
    assert await continuous.claim_due_cycle(db_session, goal.id, due) is run
    test_agent.is_active = False

    assert (await continuous.advance(db_session, goal, run, due))["step"] == "needs_attention"
    assert (await continuous.advance(db_session, goal, run, due))["step"] == "needs_attention"

    reservation = await db_session.scalar(select(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.discovery_run_id == run.id,
    ))
    assert reservation.status == "settled"
    assert reservation.settlement_reason == "needs_attention"
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id,
        OrchestrationAction.action_type == "create_delegation_task",
    )) == 0


async def test_discovery_dispatch_replays_one_task_session_and_reservation(
    db_session, test_project, test_agent, monkeypatch,
):
    goal, run = await discovery_ready(db_session, test_project, test_agent)
    service = OrchestrationService()

    async def baseline_ready(*_args, **_kwargs):
        return None

    monkeypatch.setattr(service, "_ensure_baseline_processes_ready", baseline_ready)
    await service.start_run(db_session, test_project.id, goal.id, actor="human:test")
    due = datetime.fromisoformat(goal.continuous_state["next_due_at"])
    continuous = OrchestrationContinuousService(service)
    assert await continuous.claim_due_cycle(db_session, goal.id, due) is run
    assert (await continuous.advance(db_session, goal, run, due))["step"] == "discovery_running"
    assert (await continuous.advance(db_session, goal, run, due))["step"] == "discovery_running"

    discovery = run.plan_state["discovery"]
    assert discovery["source_task_id"] == discovery["active_task_id"]
    assert discovery["active_session_id"]
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id,
        OrchestrationAction.action_type == "create_delegation_task",
    )) == 1
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.discovery_run_id == run.id,
    )) == 1
    session = await db_session.get(Session, uuid.UUID(discovery["active_session_id"]))
    assert session.metadata_["_run_config"]["timeout"] == 300
    assert session.metadata_["_run_config"]["max_tokens"] == 100


async def test_discovery_waits_with_capacity_wait_and_reserves_no_source_budget_when_root_full(
    db_session, test_project, test_agent, monkeypatch,
):
    goal, run = await discovery_ready(db_session, test_project, test_agent)
    service = OrchestrationService()

    async def baseline_ready(*_args, **_kwargs):
        return None

    monkeypatch.setattr(service, "_ensure_baseline_processes_ready", baseline_ready)
    await service.start_run(db_session, test_project.id, goal.id, actor="human:test")
    due = datetime.fromisoformat(goal.continuous_state["next_due_at"])
    continuous = OrchestrationContinuousService(service)
    assert await continuous.claim_due_cycle(db_session, goal.id, due) is run
    reserved = await db_session.scalar(select(func.count()).select_from(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.discovery_run_id == run.id))
    busy = []
    for _ in range(2):  # fill both root slots with running work of this root
        busy.append(Task(project_id=test_project.id, title="busy", status="in_progress",
                            metadata_={"orchestration": {"run_id": str(run.id)}}))
    db_session.add_all(busy)
    await db_session.flush()
    assert await continuous.advance(db_session, goal, run, due) == {"step": "waiting", "reason": "capacity_wait"}
    # the gate adds no reservation (note: claim_due_cycle already reserved the source budget before advance)
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.discovery_run_id == run.id)) == reserved
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == "create_delegation_task")) == 0
    assert not (run.plan_state or {}).get("discovery", {}).get("active_task_id")
    for task in busy:  # a freed slot lets the same cycle dispatch, still on one reservation
        task.status = "done"
    await db_session.flush()
    assert (await continuous.advance(db_session, goal, run, due))["step"] == "discovery_running"
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.discovery_run_id == run.id)) == 1


async def dispatched_discovery_source(db_session, test_project, test_agent, monkeypatch, **overrides):
    goal, run = await discovery_ready(db_session, test_project, test_agent, **overrides)
    service = OrchestrationService()

    async def baseline_ready(*_args, **_kwargs):
        return None

    monkeypatch.setattr(service, "_ensure_baseline_processes_ready", baseline_ready)
    await service.start_run(db_session, test_project.id, goal.id, actor="human:test")
    due = datetime.fromisoformat(goal.continuous_state["next_due_at"])
    continuous = OrchestrationContinuousService(service)
    assert await continuous.claim_due_cycle(db_session, goal.id, due) is run
    assert (await continuous.advance(db_session, goal, run, due))["step"] == "discovery_running"
    discovery = run.plan_state["discovery"]
    return goal, run, await db_session.get(Task, uuid.UUID(discovery["active_task_id"])), await db_session.get(
        Session, uuid.UUID(discovery["active_session_id"])
    ), continuous


async def test_discovery_terminal_output_is_consumed_once_and_frozen(
    db_session, test_project, test_agent, monkeypatch,
):
    goal, run, task, session, continuous = await dispatched_discovery_source(
        db_session, test_project, test_agent, monkeypatch,
    )
    task.status = "done"
    session.status = "completed"
    session.output = '{"schema_version":1,"candidates":[]}'
    session.metadata_ = {**session.metadata_, "token_count_in": 0, "token_count_out": 0,
                         "token_usage_complete": True, "_roadmap_elapsed_seconds": 0}
    await continuous.advance(db_session, goal, run)
    await continuous.advance(db_session, goal, run)

    assert run.plan_state["discovery"]["accepted_batch"] == {"schema_version": 1, "candidates": []}
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id,
        OrchestrationAction.action_type == "report_consumed",
    )) == 1


async def test_discovery_batch_creates_candidate_and_child(db_session, test_project, test_agent, monkeypatch):
    goal, run, task, session, continuous = await dispatched_discovery_source(
        db_session, test_project, test_agent, monkeypatch,
    )
    task.status = "done"
    session.status = "completed"
    session.output = '{"schema_version":1,"candidates":[{"origin_key":"source:1","objective":"Inspect record","source_refs":["record:1"]}]}'
    session.metadata_ = {**session.metadata_, "token_count_in": 0, "token_count_out": 0,
                         "token_usage_complete": True, "_roadmap_elapsed_seconds": 0}

    result = await continuous.advance(db_session, goal, run)

    assert result["outcome"] == "children_created"
    candidate = await db_session.scalar(select(OrchestrationContinuousCandidate).where(
        OrchestrationContinuousCandidate.parent_goal_id == goal.id,
    ))
    child = await db_session.get(OrchestrationGoal, candidate.child_goal_id)
    assert candidate.origin_key == child.continuous_origin_key == "source:1"
    assert child.objective == goal.continuous_policy["child_template"]["objective"]
    assert candidate.snapshot["policy"] == run.plan_state["continuous_policy"]
    assert child.orchestrator_context["continuous"]["discovery_provenance"]["candidate"]["objective"] == "Inspect record"


async def test_discovery_max_length_origin_uses_candidate_keys_for_release_and_event(
    db_session, test_project, test_agent, monkeypatch,
):
    goal, run, task, session, continuous = await dispatched_discovery_source(
        db_session, test_project, test_agent, monkeypatch,
    )
    origin_key = "o" * 255
    task.status = "done"
    session.status = "completed"
    session.output = (
        '{"schema_version":1,"candidates":[{"origin_key":"' + origin_key
        + '","objective":"Inspect record","source_refs":["record:1"]}]}'
    )
    session.metadata_ = {**session.metadata_, "token_count_in": 0, "token_count_out": 0,
                         "token_usage_complete": True, "_roadmap_elapsed_seconds": 0}

    assert (await continuous.advance(db_session, goal, run))["outcome"] == "children_created"

    candidate = await db_session.scalar(select(OrchestrationContinuousCandidate).where(
        OrchestrationContinuousCandidate.parent_goal_id == goal.id,
    ))
    release = await db_session.scalar(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id,
        OrchestrationAction.action_type == "release_continuous_child",
    ))
    event = await db_session.scalar(select(EventLog).where(
        EventLog.event_type == "orchestration.continuous_child_released",
        EventLog.payload["child_goal_id"].as_string() == str(candidate.child_goal_id),
    ))
    assert candidate.origin_key == origin_key
    assert origin_key == candidate.snapshot["parent_contract"]["origin_key"]
    assert str(candidate.id) in release.idempotency_key and len(release.idempotency_key) <= 255
    assert str(candidate.id) in event.dedup_key and len(event.dedup_key) <= 255


async def test_discovery_duplicate_across_cycles_creates_one_candidate_and_child(
    db_session, test_project, test_agent, monkeypatch,
):
    goal, run, task, session, continuous = await dispatched_discovery_source(
        db_session, test_project, test_agent, monkeypatch,
    )
    task.status = "done"
    session.status = "completed"
    session.output = '{"schema_version":1,"candidates":[{"origin_key":"source:1","objective":"Inspect record","source_refs":["record:1"]}]}'
    session.metadata_ = {**session.metadata_, "token_count_in": 0, "token_count_out": 0,
                         "token_usage_complete": True, "_roadmap_elapsed_seconds": 0}
    await continuous.advance(db_session, goal, run)

    successor = await OrchestrationService().get_active_run_for_goal(db_session, test_project.id, goal.id)
    due = datetime.fromisoformat(goal.continuous_state["next_due_at"])
    assert await continuous.claim_due_cycle(db_session, goal.id, due) is successor
    await continuous.advance(db_session, goal, successor, due)
    discovery = successor.plan_state["discovery"]
    duplicate_task = await db_session.get(Task, uuid.UUID(discovery["active_task_id"]))
    duplicate_session = await db_session.get(Session, uuid.UUID(discovery["active_session_id"]))
    duplicate_task.status = "done"
    duplicate_session.status = "completed"
    duplicate_session.output = session.output
    duplicate_session.metadata_ = {**duplicate_session.metadata_, "token_count_in": 0, "token_count_out": 0,
                                   "token_usage_complete": True, "_roadmap_elapsed_seconds": 0}
    await continuous.advance(db_session, goal, successor, due)

    assert await db_session.scalar(select(func.count()).select_from(OrchestrationContinuousCandidate).where(
        OrchestrationContinuousCandidate.parent_goal_id == goal.id,
    )) == 1
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationGoal).where(
        OrchestrationGoal.parent_goal_id == goal.id,
    )) == 1


async def test_discovery_budget_wait_queues_candidate_without_child_reservation(
    db_session, test_project, test_agent, monkeypatch,
):
    goal, run, task, session, continuous = await dispatched_discovery_source(
        db_session, test_project, test_agent, monkeypatch,
    )
    task.status = "done"
    session.status = "completed"
    session.output = '{"schema_version":1,"candidates":[{"origin_key":"source:1","objective":"Inspect record","source_refs":["record:1"]}]}'
    session.metadata_ = {**session.metadata_, "token_count_in": 0, "token_count_out": 0,
                         "token_usage_complete": True, "_roadmap_elapsed_seconds": 0}

    async def budget_wait(*_args, **_kwargs):
        raise HTTPException(status_code=409, detail="continuous_budget_wait")

    monkeypatch.setattr(OrchestrationBudgetService, "reserve_continuous_child", budget_wait)
    assert (await continuous.advance(db_session, goal, run))["outcome"] == "backlogged"

    candidate = await db_session.scalar(select(OrchestrationContinuousCandidate).where(
        OrchestrationContinuousCandidate.parent_goal_id == goal.id,
    ))
    assert candidate.child_goal_id is None
    assert goal.continuous_state["pending_candidates"] == [{"candidate_id": str(candidate.id), "reason": "budget_wait"}]
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == "release_continuous_child",
    )) == 0


async def test_discovery_two_candidates_create_then_queue_in_source_order(
    db_session, test_project, test_agent, monkeypatch,
):
    goal, run, task, session, continuous = await dispatched_discovery_source(
        db_session, test_project, test_agent, monkeypatch, max_active_cases=1, max_backlog=1,
    )
    task.status = "done"
    session.status = "completed"
    session.output = ('{"schema_version":1,"candidates":['
                      '{"origin_key":"source:1","objective":"First","source_refs":["record:1"]},'
                      '{"origin_key":"source:2","objective":"Second","source_refs":["record:2"]}]}')
    session.metadata_ = {**session.metadata_, "token_count_in": 0, "token_count_out": 0,
                         "token_usage_complete": True, "_roadmap_elapsed_seconds": 0}

    assert (await continuous.advance(db_session, goal, run))["outcome"] == "children_created"

    rows = list((await db_session.scalars(select(OrchestrationContinuousCandidate).where(
        OrchestrationContinuousCandidate.parent_goal_id == goal.id,
    ).order_by(OrchestrationContinuousCandidate.position))).all())
    assert [row.origin_key for row in rows] == ["source:1", "source:2"]
    assert rows[0].child_goal_id is not None and rows[1].child_goal_id is None
    assert goal.continuous_state["pending_candidates"] == [{"candidate_id": str(rows[1].id), "reason": "capacity"}]


async def test_discovery_dedupes_existing_child_and_legacy_pending_origin(
    db_session, test_project, test_agent, monkeypatch,
):
    goal, run, task, session, continuous = await dispatched_discovery_source(
        db_session, test_project, test_agent, monkeypatch,
    )
    policy = run.plan_state["continuous_policy"]
    await continuous._release_direct_child(  # pylint: disable=protected-access
        db_session, goal, run, policy, policy["child_template"], datetime.now(timezone.utc),
    )
    goal.continuous_state = {**goal.continuous_state, "pending_candidates": [{"origin_key": "legacy:1"}]}
    task.status = "done"
    session.status = "completed"
    session.output = ('{"schema_version":1,"candidates":['
                      f'{{"origin_key":"{run.cycle_key}","objective":"Duplicate child","source_refs":["record:1"]}},'
                      '{"origin_key":"legacy:1","objective":"Duplicate pending","source_refs":["record:2"]}]}')
    session.metadata_ = {**session.metadata_, "token_count_in": 0, "token_count_out": 0,
                         "token_usage_complete": True, "_roadmap_elapsed_seconds": 0}

    assert (await continuous.advance(db_session, goal, run))["outcome"] == "no_action"
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationContinuousCandidate).where(
        OrchestrationContinuousCandidate.parent_goal_id == goal.id,
    )) == 0


async def test_discovery_two_candidate_collisions_leave_survivor_for_fan_out(
    db_session, test_project, test_agent, monkeypatch,
):
    source = {**discovery_policy()["discovery_source"], "max_candidates": 3}
    goal, run, task, session, continuous = await dispatched_discovery_source(
        db_session, test_project, test_agent, monkeypatch, discovery_source=source,
    )
    task.status = "done"
    session.status = "completed"
    session.output = ('{"schema_version":1,"candidates":['
                      '{"origin_key":"source:1","objective":"First","source_refs":["record:1"]},'
                      '{"origin_key":"source:2","objective":"Second","source_refs":["record:2"]},'
                      '{"origin_key":"source:3","objective":"Third","source_refs":["record:3"]}]}')
    session.metadata_ = {**session.metadata_, "token_count_in": 0, "token_count_out": 0,
                         "token_usage_complete": True, "_roadmap_elapsed_seconds": 0}
    original_flush = db_session.flush
    original_represented = continuous._represented_discovery_origins  # pylint: disable=protected-access
    collisions = 0
    reloads = 0

    async def collision_twice(*args, **kwargs):
        nonlocal collisions
        if collisions < 2 and any(isinstance(row, OrchestrationContinuousCandidate) for row in db_session.new):
            collisions += 1
            raise IntegrityError("insert candidate", {}, Exception("unique"))
        return await original_flush(*args, **kwargs)

    async def represented_after_collision(*args, **kwargs):
        nonlocal reloads
        reloads += 1
        origins = await original_represented(*args, **kwargs)
        if reloads == 2:
            return origins | {"source:1"}
        if reloads == 3:
            return origins | {"source:1", "source:2"}
        return origins

    monkeypatch.setattr(db_session, "flush", collision_twice)
    monkeypatch.setattr(continuous, "_represented_discovery_origins", represented_after_collision)
    assert (await continuous.advance(db_session, goal, run))["outcome"] == "children_created"
    assert (collisions, reloads) == (2, 3)
    assert (await db_session.scalar(select(OrchestrationContinuousCandidate.origin_key).where(
        OrchestrationContinuousCandidate.parent_goal_id == goal.id,
    ))) == "source:3"


async def test_discovery_overflow_rejects_fresh_batch_before_candidate_insertion(
    db_session, test_project, test_agent, monkeypatch,
):
    goal, run, task, session, continuous = await dispatched_discovery_source(
        db_session, test_project, test_agent, monkeypatch, max_active_cases=1, max_backlog=1,
    )
    policy = run.plan_state["continuous_policy"]
    await continuous._release_direct_child(  # pylint: disable=protected-access
        db_session, goal, run, policy, policy["child_template"], datetime.now(timezone.utc),
    )
    task.status = "done"
    session.status = "completed"
    session.output = ('{"schema_version":1,"candidates":['
                      '{"origin_key":"source:1","objective":"Second","source_refs":["record:1"]},'
                      '{"origin_key":"source:2","objective":"Third","source_refs":["record:2"]}]}')
    session.metadata_ = {**session.metadata_, "token_count_in": 0, "token_count_out": 0,
                         "token_usage_complete": True, "_roadmap_elapsed_seconds": 0}

    assert (await continuous.advance(db_session, goal, run))["step"] == "needs_attention"
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationContinuousCandidate).where(
        OrchestrationContinuousCandidate.parent_goal_id == goal.id,
    )) == 0
    assert any(blocker["kind"] == "continuous_backlog_overflow" for blocker in run.active_blockers)


async def _discovery_triggers(db, run, blocker_code):
    return [row for row in (await db.scalars(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == "recovery_trigger",
    ))).all() if row.request["blocker_code"] == blocker_code]


async def test_discovery_invalid_output_starts_one_recovery_lineage_not_a_repair(
    db_session, test_project, test_agent, monkeypatch,
):
    """a repair delegation is a corrective fix -> one recovery trigger, no auto-repair."""
    goal, run, task, session, continuous = await dispatched_discovery_source(
        db_session, test_project, test_agent, monkeypatch,
    )
    task.status = "done"
    session.status = "completed"
    session.output = "x" * 20_000
    session.metadata_ = {**session.metadata_, "token_count_in": 0, "token_count_out": 0,
                         "token_usage_complete": True, "_roadmap_elapsed_seconds": 0}

    result = await continuous.advance(db_session, goal, run)

    assert result["step"] == "waiting_discovery_recovery"
    assert run.plan_state["discovery"].get("recovery") is None
    assert await db_session.scalar(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == "create_delegation_task",
        OrchestrationAction.request["parent_task_id"].as_string() == str(task.id),
    )) is None
    triggers = await _discovery_triggers(db_session, run, "continuous_discovery_output_invalid")
    assert len(triggers) == 1 and len(triggers[0].request["summary"]) < 4_001


async def test_discovery_failed_source_starts_one_recovery_lineage_not_a_retry(
    db_session, test_project, test_agent, monkeypatch,
):
    """a new source attempt needs approve_fix -> one trigger, no retry/session/budget."""
    goal, run, task, session, continuous = await dispatched_discovery_source(
        db_session, test_project, test_agent, monkeypatch,
    )
    task.status = "failed"
    session.status = "failed"
    session.metadata_ = {**session.metadata_, "token_count_in": 0, "token_count_out": 0,
                         "token_usage_complete": True, "_roadmap_elapsed_seconds": 0,
                         "attempt": {"effect_state": "started"}}

    assert (await continuous.advance(db_session, goal, run))["step"] == "waiting_discovery_recovery"
    assert (await continuous.advance(db_session, goal, run))["step"] == "waiting_discovery_recovery"  # replay

    assert await db_session.scalar(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == "retry_task",
    )) is None
    assert await db_session.scalar(select(func.count(Session.id)).where(Session.task_id == task.id)) == 1
    assert len(await _discovery_triggers(db_session, run, "continuous_discovery_source_failed")) == 1


@pytest.mark.parametrize(("limiting_budget", "expired", "can_reactivate"), [
    ("rolling", False, True), ("rolling", True, True),
    ("rolling", False, False), ("rolling", True, False), ("total", False, False),
])
async def test_legacy_discovery_reservation_reactivation_respects_budget_headroom(
    db_session, test_project, test_agent, monkeypatch, limiting_budget, expired, can_reactivate,
):
    goal, run, task, session, _continuous = await dispatched_discovery_source(
        db_session, test_project, test_agent, monkeypatch,
    )
    task.status = session.status = "failed"
    session.metadata_ = {**session.metadata_, "token_count_in": 20, "token_count_out": 0,
                         "token_usage_complete": True, "_roadmap_elapsed_seconds": 0}
    budget = OrchestrationBudgetService()
    reservation = await db_session.scalar(select(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.discovery_run_id == run.id,
    ))
    await budget.settle_discovery_source(db_session, reservation, run, reason="needs_attention")
    now = datetime.now(timezone.utc)
    if expired:
        reservation.settled_at = now - timedelta(hours=2)
    source = deepcopy(run.plan_state["continuous_policy"]["discovery_source"])
    if limiting_budget == "rolling":
        source["budget"]["rolling"]["limits"]["max_tokens"] = "100" if can_reactivate else "80"
    else:
        goal.budget = {"caps": {**goal.budget["caps"], "max_tokens": "80"}}
    await db_session.flush()

    if can_reactivate:
        assert await budget.reactivate_discovery_source(db_session, goal, run, source, now) is reservation
        assert reservation.status == "active" and reservation.settled_spend == {}
    else:
        with pytest.raises(HTTPException) as exc:
            await budget.reactivate_discovery_source(db_session, goal, run, source, now)
        assert exc.value.detail == "continuous_discovery_budget_wait"
        assert reservation.status == "settled" and reservation.settled_spend["max_tokens"] == "20"


async def test_discovery_repeated_invalid_output_keeps_one_lineage_without_reassignment(
    db_session, test_project, test_agent, monkeypatch,
):
    goal, run, task, session, continuous = await dispatched_discovery_source(
        db_session, test_project, test_agent, monkeypatch,
    )
    task.status = "done"
    session.status = "completed"
    session.output = "not json"
    session.metadata_ = {**session.metadata_, "token_count_in": 0, "token_count_out": 0,
                         "token_usage_complete": True, "_roadmap_elapsed_seconds": 0}
    assert (await continuous.advance(db_session, goal, run))["step"] == "waiting_discovery_recovery"
    assert (await continuous.advance(db_session, goal, run))["step"] == "waiting_discovery_recovery"

    assert len(await _discovery_triggers(db_session, run, "continuous_discovery_output_invalid")) == 1
    assert not await db_session.scalar(select(OrchestrationAction.id).where(
        OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == "reassign_task",
    ))
    assert not await db_session.scalar(select(OrchestrationAction.id).where(
        OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == "create_delegation_task",
        OrchestrationAction.request["parent_task_id"].as_string() == str(task.id),
    ))


async def test_completed_discovery_batch_replay_is_stable(db_session, test_project, test_agent, monkeypatch):
    goal, run, task, session, continuous = await dispatched_discovery_source(
        db_session, test_project, test_agent, monkeypatch,
    )
    task.status = "done"
    session.status = "completed"
    session.output = '{"schema_version":1,"candidates":[]}'
    session.metadata_ = {**session.metadata_, "token_count_in": 0, "token_count_out": 0,
                         "token_usage_complete": True, "_roadmap_elapsed_seconds": 0}
    await continuous.advance(db_session, goal, run)
    run.plan_state["discovery"]["accepted_batch"] = None
    run.plan_state["discovery"]["active_session_id"] = str(uuid.uuid4())

    assert (await continuous.advance(db_session, goal, run))["outcome"] == "no_action"


async def test_discovery_fresh_terminal_output_rejects_missing_expected_session(
    db_session, test_project, test_agent, monkeypatch,
):
    goal, run, task, session, continuous = await dispatched_discovery_source(
        db_session, test_project, test_agent, monkeypatch,
    )
    task.status = "done"
    session.status = "completed"
    run.plan_state["discovery"]["active_session_id"] = str(uuid.uuid4())

    # a non-canonical terminal session is surfaced as attention, never an every-tick raise.
    assert (await continuous.advance(db_session, goal, run))["step"] == "needs_attention"
    blocker = next(b for b in run.active_blockers if b["kind"] == "continuous_discovery_source_execution_failed")
    assert "Terminal task session is not canonical" in blocker["error"]


async def test_discovery_missing_reservation_is_stable_budget_integrity(
    db_session, test_project, test_agent, monkeypatch,
):
    goal, run, task, session, continuous = await dispatched_discovery_source(
        db_session, test_project, test_agent, monkeypatch,
    )
    task.status = session.status = "failed"
    await db_session.execute(delete(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.discovery_run_id == run.id,
    ))
    await db_session.flush()

    assert (await continuous.advance(db_session, goal, run))["step"] == "needs_attention"
    assert (await continuous.advance(db_session, goal, run))["step"] == "needs_attention"
    assert any(blocker["kind"] == "budget_integrity" for blocker in run.active_blockers)


def test_discovery_contract_is_strict():
    policy = OrchestrationContinuousPolicy.model_validate(discovery_policy())
    assert policy.discovery_source.instructions == "List records"
    assert parse_discovery_batch(' \n{"schema_version":1,"candidates":[]}\n ', 2).candidates == []


@pytest.mark.parametrize(
    "raw",
    [
        '```json\n{"schema_version":1,"candidates":[]}\n```',
        '{"schema_version":1,"candidates":[],"x":1}',
        '{"schema_version":true,"candidates":[]}',
        '{"schema_version":1.0,"candidates":[]}',
        ('{"schema_version":1,"candidates":[{"origin_key":"one","objective":"First",'
         '"source_refs":["a"]},{"origin_key":"one","objective":"Second","source_refs":["b"]}]}'),
        ('{"schema_version":1,"candidates":[{"origin_key":"one","objective":"First",'
         '"source_refs":["a"]},{"origin_key":"two","objective":"Second","source_refs":["b"]}]}'),
        '{"schema_version":1,"candidates":[{"origin_key":" ","objective":"First","source_refs":["a"]}]}',
        ('{"schema_version":1,"candidates":[{"origin_key":"one","objective":"' + "x" * 4001
         + '","source_refs":["a"]}]}'),
    ],
)
def test_discovery_batch_rejects_invalid_shape_or_candidates(raw):
    with pytest.raises(ValueError):
        parse_discovery_batch(raw, 1)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("max_candidates", True),
        ("max_candidates", "2"),
        ("timeout_seconds", 0),
        ("timeout_seconds", "300"),
    ],
)
def test_discovery_source_rejects_invalid_strict_integers(path, value):
    policy = discovery_policy()
    policy["discovery_source"][path] = value
    with pytest.raises(ValidationError):
        OrchestrationContinuousPolicy.model_validate(policy)


@pytest.mark.parametrize(
    "policy",
    [
        direct_policy(adapter_filter=None),
        direct_policy(discovery_source=discovery_policy()["discovery_source"]),
        discovery_policy(adapter_filter={"adapter_type": "schedule", "enabled": True}),
        discovery_policy(discovery_source=None),
        discovery_policy(per_case_budget={"max_tokens": "100", "max_turns": "2"}),
        discovery_policy(discovery_source={
            **discovery_policy()["discovery_source"],
            "budget": {
                **discovery_policy()["discovery_source"]["budget"],
                "per_cycle": {"max_tokens": "100", "max_turns": "2"},
            },
        }),
    ],
)
def test_discovery_policy_rejects_mode_source_or_budget_dimension_mismatch(policy):
    with pytest.raises(ValidationError):
        OrchestrationContinuousPolicy.model_validate(policy)


def test_continuous_policy_normalizes_direct_contract():
    policy = OrchestrationContinuousPolicy.model_validate(direct_policy())
    assert policy.activation.timezone == "UTC"
    assert policy.per_case_budget == {"max_tokens": "100", "max_turns": "2", "max_hours": "0.5"}
    assert policy.model_dump(mode="json")["cycle_mode"] == "direct"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("activation", {"cron": "not cron", "timezone": "UTC", "missed_slots": "coalesce"}),
        ("activation", {"cron": "0 * * * *", "timezone": "Mars/Olympus", "missed_slots": "coalesce"}),
        ("activation", {"cron": "0 * * * *", "timezone": "UTC", "missed_slots": "every"}),
        ("cycle_mode", "discovery"),
        ("per_case_budget", {"max_tokens": -1}),
        ("max_active_cases", 0),
        ("max_backlog", 0),
        ("response_target_seconds", True),
        ("max_active_cases", True),
        ("max_backlog", True),
        ("rolling_budget", {
            "window_seconds": True,
            "limits": {"max_tokens": "500", "max_turns": "10", "max_hours": "2"},
        }),
        ("stop_condition", {"mode": "max_cycles", "max_cycles": True}),
    ],
)
def test_continuous_policy_rejects_unsupported_or_unbounded_values(field, value):
    with pytest.raises(ValidationError):
        OrchestrationContinuousPolicy.model_validate(direct_policy(**{field: value}))


@pytest.mark.parametrize("amount", ["1e1000000", "1e-1000000"])
def test_continuous_policy_rejects_unsafe_budget_exponents(amount):
    with pytest.raises(ValidationError):
        OrchestrationContinuousPolicy.model_validate(
            direct_policy(per_case_budget={"max_tokens": amount, "max_turns": "2", "max_hours": "0.5"})
        )


@pytest.mark.asyncio
async def test_cycle_and_origin_keys_are_unique_per_parent(db_session, test_project):
    parent = OrchestrationGoal(
        project_id=test_project.id, objective="Continuous parent", original_request="Continuous parent",
        goal_type="continuous", continuous_policy={"version": 1}, continuous_state={"policy_version": 1},
    )
    db_session.add(parent)
    await db_session.flush()
    db_session.add(OrchestrationRun(
        goal_id=parent.id, phase="completed", status="completed", cycle_key="2026-09-06T10:00:00Z",
    ))
    await db_session.flush()
    db_session.add(OrchestrationRun(
        goal_id=parent.id, phase="completed", status="completed", cycle_key="2026-09-06T10:00:00Z",
    ))
    with pytest.raises(IntegrityError):
        await db_session.flush()


@pytest.mark.asyncio
async def test_continuous_child_origin_is_unique_across_cycles(db_session, test_project):
    parent = OrchestrationGoal(
        project_id=test_project.id, objective="Parent", original_request="Parent", goal_type="continuous",
    )
    db_session.add(parent)
    await db_session.flush()
    db_session.add(OrchestrationGoal(
        project_id=test_project.id, objective="First", original_request="First",
        parent_goal_id=parent.id, continuous_origin_key="slot:one",
        parent_contract_snapshot={}, goal_delta={},
    ))
    await db_session.flush()
    db_session.add(OrchestrationGoal(
        project_id=test_project.id, objective="Duplicate", original_request="Duplicate",
        parent_goal_id=parent.id, continuous_origin_key="slot:one",
        parent_contract_snapshot={}, goal_delta={},
    ))
    with pytest.raises(IntegrityError):
        await db_session.flush()


async def continuous_ready(db_session, test_project):
    goal = OrchestrationGoal(
        project_id=test_project.id, objective="Watch scheduled cases", original_request="Watch scheduled cases",
        success_criteria=[{"key": "responded", "description": "Each case is verified"}],
        constraints={}, budget={"caps": {"max_tokens": 5000, "max_turns": 100, "max_hours": 20}},
        goal_type="continuous", status="active",
    )
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id, phase="ready", status="running", budget_state=goal.budget)
    db_session.add(run)
    await db_session.flush()
    policy = OrchestrationContinuousPolicy.model_validate(direct_policy())
    await OrchestrationContinuousService(OrchestrationService()).update_policy(
        db_session, test_project.id, goal.id, policy, actor="human:test",
    )
    return goal, run


async def test_policy_versions_are_snapshotted_only_when_claimed(db_session, test_project):
    goal, run = await continuous_ready(db_session, test_project)
    assert goal.continuous_policy["version"] == 1
    assert goal.continuous_state["policy_version"] == 1
    policy = OrchestrationContinuousPolicy.model_validate(direct_policy(response_target_seconds=600))
    await OrchestrationContinuousService(OrchestrationService()).update_policy(
        db_session, test_project.id, goal.id, policy, actor="human:test",
    )
    assert goal.continuous_policy["version"] == 2
    assert run.plan_state == {}


async def test_start_moves_continuous_run_to_waiting_activation_once(db_session, test_project):
    goal, run = await continuous_ready(db_session, test_project)
    service = OrchestrationService()
    first_goal, first_run = await service.start_run(db_session, test_project.id, goal.id, actor="human:test")
    first_state = deepcopy(first_goal.continuous_state)
    due = datetime.fromisoformat(first_state["next_due_at"])
    assert first_run.id == run.id and first_run.phase == "waiting_activation"
    assert due.tzinfo is not None
    second_goal, second_run = await service.start_run(db_session, test_project.id, goal.id, actor="human:test")
    assert second_run.id == first_run.id
    assert second_goal.continuous_state == first_state
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id,
        OrchestrationAction.action_type == "authorize_execution",
    )) == 1


async def test_start_rejects_missing_policy_without_authorizing(db_session, test_project):
    goal = OrchestrationGoal(
        project_id=test_project.id, objective="Continuous", original_request="Continuous", goal_type="continuous",
    )
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id, phase="ready")
    db_session.add(run)
    await db_session.flush()
    with pytest.raises(HTTPException) as exc:
        await OrchestrationService().start_run(db_session, test_project.id, goal.id, actor="human:test")
    assert exc.value.detail["conflict"] == "goal_not_runnable"
    assert run.phase == "ready"


async def test_policy_rejects_child_constraint_that_weakens_parent(db_session, test_project):
    goal, _ = await continuous_ready(db_session, test_project)
    goal.constraints = {"external_impact": False}
    policy = direct_policy()
    policy["child_template"]["constraints"] = {"external_impact": True}
    with pytest.raises(HTTPException) as exc:
        await OrchestrationContinuousService(OrchestrationService()).update_policy(
            db_session, test_project.id, goal.id,
            OrchestrationContinuousPolicy.model_validate(policy), actor="human:test",
        )
    assert exc.value.detail == "Continuous child constraint conflicts with parent: external_impact"


async def test_rolling_available_subtracts_recent_settlement_and_all_active_reservations(
    db_session, test_project,
):
    goal, _ = await continuous_ready(db_session, test_project)
    now = datetime(2026, 9, 6, 12, 0, tzinfo=timezone.utc)
    recent_child = OrchestrationGoal(project_id=test_project.id, objective="Recent", original_request="Recent")
    active_child = OrchestrationGoal(project_id=test_project.id, objective="Active", original_request="Active")
    db_session.add_all([recent_child, active_child])
    await db_session.flush()
    db_session.add_all([
        OrchestrationBudgetReservation(
            parent_goal_id=goal.id, roadmap_item_id=None, child_goal_id=recent_child.id,
            continuous_origin_key="recent", allocation={"max_tokens": "100", "max_turns": "2", "max_hours": "0.5"},
            settled_spend={"max_tokens": "80", "max_turns": "1", "max_hours": "0.25"},
            status="settled", settlement_reason="completed", settled_at=now - timedelta(minutes=10),
        ),
        OrchestrationBudgetReservation(
            parent_goal_id=goal.id, roadmap_item_id=None, child_goal_id=active_child.id,
            continuous_origin_key="active", allocation={"max_tokens": "100", "max_turns": "2", "max_hours": "0.5"},
        ),
    ])
    await db_session.flush()
    available = await OrchestrationBudgetService().continuous_available(db_session, goal, goal.continuous_policy, now)
    assert available["rolling"] == {"max_tokens": "320", "max_turns": "7", "max_hours": "1.25"}
    assert all(Decimal(available["available"][key]) <= Decimal(available["total"][key]) for key in available["available"])


async def test_old_settlement_leaves_rolling_window_but_remains_in_total_cap(db_session, test_project):
    goal, _ = await continuous_ready(db_session, test_project)
    now = datetime(2026, 9, 6, 12, 0, tzinfo=timezone.utc)
    child = OrchestrationGoal(project_id=test_project.id, objective="Old", original_request="Old")
    db_session.add(child)
    await db_session.flush()
    db_session.add(OrchestrationBudgetReservation(
        parent_goal_id=goal.id, roadmap_item_id=None, child_goal_id=child.id,
        continuous_origin_key="old", allocation={"max_tokens": "100", "max_turns": "2", "max_hours": "0.5"},
        settled_spend={"max_tokens": "90", "max_turns": "2", "max_hours": "0.5"},
        status="settled", settlement_reason="completed", settled_at=now - timedelta(hours=2),
    ))
    await db_session.flush()
    available = await OrchestrationBudgetService().continuous_available(db_session, goal, goal.continuous_policy, now)
    assert available["rolling"] == goal.continuous_policy["rolling_budget"]["limits"]
    assert available["total"]["max_tokens"] == "4910"


async def discovery_budget_context(db_session, test_project, test_agent):
    goal, parent_run = await continuous_ready(db_session, test_project)
    parent_run.status = parent_run.phase = "completed"
    await db_session.flush()
    source_run = OrchestrationRun(
        goal_id=goal.id, phase="authorized", cycle_key="discovery:one",
        plan_state={"discovery": {}}, budget_state=goal.budget,
    )
    db_session.add(source_run)
    await db_session.flush()
    source = discovery_policy()["discovery_source"]
    source["budget"]["per_cycle"] = {"max_tokens": "20", "max_turns": "2", "max_hours": "0.1"}
    source["budget"]["rolling"]["limits"] = {"max_tokens": "20", "max_turns": "2", "max_hours": "0.1"}
    task = Task(
        project_id=test_project.id, title="Discovery", status="ready", assigned_to=test_agent.id,
        metadata_={"orchestration_contract": {"orchestrator_context": {"continuous": {
            "discovery_run_id": str(source_run.id),
        }}}},
    )
    db_session.add(task)
    await db_session.flush()
    source_run.plan_state = {"discovery": {"source_task_id": str(task.id)}}
    return goal, parent_run, source_run, source, task


async def test_discovery_reservation_replays_and_source_session_uses_reserved_remaining(
    db_session, test_project, test_agent,
):
    goal, _, run, source, task = await discovery_budget_context(db_session, test_project, test_agent)
    budget = OrchestrationBudgetService()
    first = await budget.reserve_discovery_source(db_session, goal, run, source, datetime.now(timezone.utc))
    second = await budget.reserve_discovery_source(db_session, goal, run, source, datetime.now(timezone.utc))

    assert first.id == second.id and first.discovery_run_id == run.id
    limits = await SessionService()._orchestration_budget_limits(db_session, task, test_agent)
    assert limits["max_tokens"] == 20 and limits["timeout"] == 360


async def test_discovery_session_without_active_reservation_is_integrity_attention(
    db_session, test_project, test_agent,
):
    _, _, run, _, task = await discovery_budget_context(db_session, test_project, test_agent)

    with pytest.raises(SessionClaimAttention) as exc:
        await SessionService()._orchestration_budget_limits(db_session, task, test_agent)

    assert exc.value.run_id == run.id
    assert exc.value.blocker["kind"] == "budget_integrity"


@pytest.mark.parametrize("discovery", [{}, {"source_task_id": str(uuid.uuid4())}])
async def test_discovery_session_requires_matching_durable_source_task(
    db_session, test_project, test_agent, discovery,
):
    goal, _, run, source, task = await discovery_budget_context(db_session, test_project, test_agent)
    await OrchestrationBudgetService().reserve_discovery_source(
        db_session, goal, run, source, datetime.now(timezone.utc),
    )
    run.plan_state = {"discovery": discovery}

    with pytest.raises(SessionClaimAttention) as exc:
        await SessionService()._orchestration_budget_limits(db_session, task, test_agent)

    assert exc.value.run_id == run.id
    assert exc.value.blocker["kind"] == "budget_integrity"


async def test_discovery_source_task_without_contract_cannot_fall_back_to_parent_budget(
    db_session, test_project, test_agent,
):
    goal, _, run, source, task = await discovery_budget_context(db_session, test_project, test_agent)
    await OrchestrationBudgetService().reserve_discovery_source(
        db_session, goal, run, source, datetime.now(timezone.utc),
    )
    db_session.add(OrchestrationAction(
        run_id=run.id, idempotency_key="source-contract", action_type="create_delegation_task",
        request={}, status="completed", target_type="task", target_id=task.id,
    ))
    task.metadata_ = {}
    await db_session.flush()

    with pytest.raises(SessionClaimAttention) as exc:
        await SessionService()._orchestration_budget_limits(db_session, task, test_agent)

    assert exc.value.run_id == run.id
    assert exc.value.blocker["kind"] == "budget_integrity"


async def test_discovery_settlement_without_reservation_is_integrity_attention(
    db_session, test_project, test_agent,
):
    _, _, run, _, _ = await discovery_budget_context(db_session, test_project, test_agent)

    with pytest.raises(SessionClaimAttention) as exc:
        await OrchestrationBudgetService().settle_discovery_source(db_session, None, run, reason="needs_attention")

    assert exc.value.run_id == run.id
    assert exc.value.blocker["kind"] == "budget_integrity"


async def test_discovery_settlement_retains_allocation_for_unknown_cancelled_spend(
    db_session, test_project, test_agent,
):
    goal, _, run, source, task = await discovery_budget_context(db_session, test_project, test_agent)
    reservation = await OrchestrationBudgetService().reserve_discovery_source(
        db_session, goal, run, source, datetime.now(timezone.utc),
    )
    db_session.add(Session(
        agent_id=test_agent.id, task_id=task.id, project_id=test_project.id, adapter_type="api", status="cancelled",
        metadata_={"token_usage_complete": False},
    ))
    await db_session.flush()

    settled = await OrchestrationBudgetService().settle_discovery_source(
        db_session, reservation, run, reason="cancelled",
    )
    assert settled.measurement_complete is False
    assert settled.settled_spend == reservation.allocation


async def test_discovery_tasks_are_excluded_from_parent_direct_spend(db_session, test_project, test_agent):
    goal, _, run, source, task = await discovery_budget_context(db_session, test_project, test_agent)
    reservation = await OrchestrationBudgetService().reserve_discovery_source(
        db_session, goal, run, source, datetime.now(timezone.utc),
    )
    db_session.add_all([
        OrchestrationAction(
            run_id=run.id, idempotency_key="source-task", action_type="create_delegation_task",
            request={}, status="completed", target_type="task", target_id=task.id,
        ),
            Session(
                agent_id=test_agent.id, task_id=task.id, project_id=test_project.id, adapter_type="api",
                status="completed",
            metadata_={"token_count_in": 3, "token_count_out": 2, "token_usage_complete": True,
                       "_roadmap_elapsed_seconds": 0, "_roadmap_turn_count": 1},
        ),
    ])
    await db_session.flush()
    await OrchestrationBudgetService().settle_discovery_source(db_session, reservation, run, reason="completed")

    remaining = await OrchestrationBudgetService().remaining(db_session, goal)
    assert remaining["direct_spend"]["max_tokens"] == "0"
    assert remaining["settled_child_spend"]["max_tokens"] == "5"


async def test_continuous_child_session_claim_is_limited_by_its_reservation(db_session, test_project, test_agent):
    goal, _ = await continuous_ready(db_session, test_project)
    origin_key = "2026-09-06T12:00:00Z"
    child = OrchestrationGoal(
        id=uuid.uuid4(), project_id=test_project.id, objective="Budgeted child", original_request="Budgeted child",
        goal_type="outcome", parent_goal_id=goal.id, continuous_origin_key=origin_key,
        parent_contract_snapshot={}, goal_delta={}, budget={"caps": {"max_tokens": "100", "max_turns": "2", "max_hours": "0.5"}},
    )
    child_run = OrchestrationRun(id=uuid.uuid4(), goal_id=child.id, phase="authorized", budget_state=child.budget)
    task = Task(project_id=test_project.id, title="Budgeted work", status="ready", assigned_to=test_agent.id)
    db_session.add_all([child, child_run, task])
    await db_session.flush()
    db_session.add_all([
        OrchestrationBudgetReservation(
            parent_goal_id=goal.id, roadmap_item_id=None, child_goal_id=child.id,
            continuous_origin_key=origin_key,
            allocation={"max_tokens": "100", "max_turns": "2", "max_hours": "0.5"},
        ),
        OrchestrationAction(
            run_id=child_run.id, idempotency_key=f"test:continuous-task:{task.id}",
            action_type="create_delegation_task", request={}, status="completed", target_type="task", target_id=task.id,
        ),
    ])
    await db_session.flush()
    assert await OrchestrationBudgetService().owning_continuous_run(db_session, task) == child_run
    session = await SessionService().create(db_session, SessionCreate(
        agent_id=test_agent.id, task_id=task.id, project_id=test_project.id, max_tokens=1000,
    ))
    assert session.metadata_["_run_config"]["max_tokens"] == 100


async def test_continuous_cli_budget_claim_fails_closed_with_deduplicated_blocker(
    db_session, test_project, test_agent,
):
    goal, _ = await continuous_ready(db_session, test_project)
    child = OrchestrationGoal(
        project_id=test_project.id, objective="CLI child", original_request="CLI child", goal_type="outcome",
        parent_goal_id=goal.id, continuous_origin_key="cli", parent_contract_snapshot={}, goal_delta={},
        budget={"caps": {"max_tokens": "100", "max_turns": "2", "max_hours": "0.5"}},
    )
    task = Task(project_id=test_project.id, title="CLI work", status="ready", assigned_to=test_agent.id)
    db_session.add(child)
    await db_session.flush()
    child_run = OrchestrationRun(goal_id=child.id, phase="authorized", budget_state=child.budget)
    db_session.add_all([child_run, task])
    await db_session.flush()
    db_session.add_all([
        OrchestrationBudgetReservation(parent_goal_id=goal.id, roadmap_item_id=None, child_goal_id=child.id,
            continuous_origin_key="cli", allocation={"max_tokens": "100", "max_turns": "2", "max_hours": "0.5"}),
        OrchestrationAction(run_id=child_run.id, idempotency_key=f"test:continuous-cli:{task.id}",
            action_type="create_delegation_task", request={}, status="completed", target_type="task", target_id=task.id),
    ])
    await db_session.flush()
    reservation = await db_session.scalar(select(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.child_goal_id == child.id,
    ))
    reservation_state = (deepcopy(reservation.allocation), reservation.status, deepcopy(reservation.settled_spend))
    test_agent.adapter_type = "cli"
    service = SessionService()
    with pytest.raises(SessionClaimAttention) as exc:
        await service.preflight_claim(db_session, task, test_agent)
    assert task.status == "ready"
    assert child_run.active_blockers == []
    assert await db_session.scalar(select(func.count()).select_from(Session).where(Session.task_id == task.id)) == 0
    assert (reservation.allocation, reservation.status, reservation.settled_spend) == reservation_state
    await SessionService.persist_claim_attention(db_session, exc.value)
    with pytest.raises(SessionClaimAttention) as exc:
        await service.create(db_session, SessionCreate(
            agent_id=test_agent.id, task_id=task.id, project_id=test_project.id,
        ))
    assert task.status == "ready"
    assert await db_session.scalar(select(func.count()).select_from(Session).where(Session.task_id == task.id)) == 0
    assert (reservation.allocation, reservation.status, reservation.settled_spend) == reservation_state
    await SessionService.persist_claim_attention(db_session, exc.value)
    session = Session(
        agent_id=test_agent.id, task_id=task.id, project_id=test_project.id, adapter_type="cli",
        status="failed", resumable=True,
        metadata_={"_run_config": {"max_tokens": 100, "timeout": 1800}, "token_count_in": 0,
                   "token_count_out": 0, "token_usage_complete": True, "_roadmap_elapsed_seconds": 0},
    )
    db_session.add(session)
    await db_session.flush()
    original_metadata = deepcopy(session.metadata_)
    with pytest.raises(SessionClaimAttention) as exc:
        await service.resume(db_session, session.id)
    await SessionService.persist_claim_attention(db_session, exc.value)
    assert session.status == "failed" and session.resumable and session.runner_task_id is None
    assert session.metadata_ == original_metadata
    assert task.status == "ready"
    assert (reservation.allocation, reservation.status, reservation.settled_spend) == reservation_state
    assert await db_session.scalar(select(func.count()).select_from(Session).where(Session.task_id == task.id)) == 1
    for _ in range(2):
        with pytest.raises(SessionClaimAttention) as exc:
            await service.create(db_session, SessionCreate(
                agent_id=test_agent.id, task_id=task.id, project_id=test_project.id,
            ))
        assert exc.value.blocker["kind"] == "budget_integrity"
        assert exc.value.blocker["reason"] == "Continuous CLI cannot enforce reserved budget dimensions: max_tokens."
        await SessionService.persist_claim_attention(db_session, exc.value)
    assert len(child_run.active_blockers) == 1
    test_agent.adapter_type = "api"
    await service.create(db_session, SessionCreate(
        agent_id=test_agent.id, task_id=task.id, project_id=test_project.id,
    ))
    scope = f"adapter-capability:{child_run.id}"
    assert not any(blocker.get("scope") == scope for blocker in child_run.active_blockers)
    OrchestrationService._upsert_active_blocker(child_run, {  # pylint: disable=protected-access
        "kind": "budget_integrity", "scope": scope, "reason": "stale adapter blocker",
    })
    await service._validate_budget_adapter(db_session, task, {"timeout": 30}, {}, "cli")  # pylint: disable=protected-access
    assert not any(blocker.get("scope") == scope for blocker in child_run.active_blockers)


async def started_continuous(db_session, test_project, *, enabled=True):
    goal, run = await continuous_ready(db_session, test_project)
    policy = OrchestrationContinuousPolicy.model_validate(
        direct_policy(adapter_filter={"adapter_type": "schedule", "enabled": enabled})
    )
    await OrchestrationContinuousService(OrchestrationService()).update_policy(
        db_session, test_project.id, goal.id, policy, actor="human:test",
    )
    await OrchestrationService().start_run(db_session, test_project.id, goal.id, actor="human:test")
    return goal, run


async def test_start_freezes_goal_contract_fingerprint(db_session, test_project):
    goal, run = await continuous_ready(db_session, test_project)
    assert run.goal_contract_fingerprint is None
    await OrchestrationService().start_run(db_session, test_project.id, goal.id, actor="human:test")
    frozen = run.goal_contract_fingerprint
    assert frozen == compute_goal_contract_fingerprint(goal)
    await OrchestrationService().start_run(db_session, test_project.id, goal.id, actor="human:test")
    assert run.goal_contract_fingerprint == frozen


async def test_g3_started_continuous_run_accepts_idempotent_execution_upgrade(db_session, test_project):
    goal, run = await continuous_ready(db_session, test_project)
    service = OrchestrationService()
    await service.start_run(db_session, test_project.id, goal.id, actor="human:test", scope="planning_only",
                            request_id=uuid.uuid4(), expected_run_id=run.id, expected_control_version=run.control_version)
    due = datetime.fromisoformat(goal.continuous_state["next_due_at"])
    continuous = OrchestrationContinuousService(service)
    assert await continuous.claim_due_cycle(db_session, goal.id, due) is None
    upgrade_request_id = uuid.uuid4()
    await service.start_run(db_session, test_project.id, goal.id, actor="human:test", scope="execution",
                            request_id=upgrade_request_id, expected_run_id=run.id, expected_control_version=run.control_version)
    assert (await continuous.claim_due_cycle(db_session, goal.id, due)).id == run.id
    await service.start_run(db_session, test_project.id, goal.id, actor="human:test", scope="execution",
                            request_id=upgrade_request_id, expected_run_id=run.id, expected_control_version=run.control_version)
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id,
        OrchestrationAction.action_type == "grant_execution",
    )) == 1


async def test_claim_is_atomic_and_coalesces_missed_slots(db_session, test_project):
    goal, run = await started_continuous(db_session, test_project)
    due = datetime.fromisoformat(goal.continuous_state["next_due_at"])
    now = due + timedelta(hours=2)
    continuous = OrchestrationContinuousService(OrchestrationService())
    first = await continuous.claim_due_cycle(db_session, goal.id, now)
    second = await continuous.claim_due_cycle(db_session, goal.id, now)
    assert first.id == run.id and first.phase == "authorized"
    assert second is None
    assert run.cycle_key == due.isoformat().replace("+00:00", "Z")
    assert datetime.fromisoformat(goal.continuous_state["next_due_at"]) > now
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id,
        OrchestrationAction.action_type == "claim_continuous_cycle",
    )) == 1


async def test_concurrent_cycle_claim_has_one_winner(concurrent_sessions):
    first_db, second_db = concurrent_sessions
    project = Project(name="Continuous claim race", config={})
    first_db.add(project)
    await first_db.flush()
    due = datetime(2026, 9, 6, 12, 0, tzinfo=timezone.utc)
    policy = {"version": 1, **direct_policy()}
    goal = OrchestrationGoal(
        project_id=project.id, objective="Claim once", original_request="Claim once",
        goal_type="continuous", budget={"caps": policy["rolling_budget"]["limits"]},
        continuous_policy=policy,
        continuous_state={"policy_version": 1, "next_due_at": due.isoformat(), "last_claimed_slot": None,
                          "pending_slots": [], "pending_candidates": [], "health": "healthy",
                          "health_reason": None, "cycles_completed": 0, "started_at": due.isoformat(),
                          "stopped_at": None},
    )
    first_db.add(goal)
    await first_db.flush()
    run = OrchestrationRun(goal_id=goal.id, phase="waiting_activation")
    first_db.add(run)
    await first_db.flush()
    first_db.add(OrchestrationAction(
        run_id=run.id, idempotency_key=f"run:{run.id}:kind:authorize_execution",
        action_type="authorize_execution", request={"actor": "human:test"},
        status="completed", target_type="run", target_id=run.id,
    ))
    await first_db.commit()

    async def claim(db):
        return await OrchestrationContinuousService(OrchestrationService()).claim_due_cycle(db, goal.id, due)

    results = await asyncio.gather(claim(first_db), claim(second_db))
    assert sum(result is not None for result in results) == 1
    assert await first_db.scalar(select(func.count()).select_from(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id,
        OrchestrationAction.action_type == "claim_continuous_cycle",
    )) == 1


async def test_empty_direct_cycle_is_no_action_and_parent_stays_active(db_session, test_project):
    goal, run = await started_continuous(db_session, test_project, enabled=False)
    continuous = OrchestrationContinuousService(OrchestrationService())
    due = datetime.fromisoformat(goal.continuous_state["next_due_at"])
    await continuous.claim_due_cycle(db_session, goal.id, due)
    result = await continuous.advance(db_session, goal, run, due)
    successor = await OrchestrationService().get_active_run_for_goal(db_session, test_project.id, goal.id)
    assert result["outcome"] == "no_action"
    assert run.status == run.phase == "completed"
    assert goal.status == "active" and successor.phase == "waiting_activation"
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationGoal).where(
        OrchestrationGoal.parent_goal_id == goal.id,
    )) == 0


async def test_direct_cycle_creates_one_reserved_outcome_and_completes_immediately(db_session, test_project):
    goal, run = await started_continuous(db_session, test_project)
    continuous = OrchestrationContinuousService(OrchestrationService())
    due = datetime.fromisoformat(goal.continuous_state["next_due_at"])
    await continuous.claim_due_cycle(db_session, goal.id, due)
    first = await continuous.advance(db_session, goal, run, due)
    second = await continuous.advance(db_session, goal, run, due)
    children = list((await db_session.scalars(select(OrchestrationGoal).where(
        OrchestrationGoal.parent_goal_id == goal.id,
    ))).all())
    reservations = list((await db_session.scalars(select(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.parent_goal_id == goal.id,
    ))).all())
    successor = await OrchestrationService().get_active_run_for_goal(db_session, test_project.id, goal.id)
    assert first["outcome"] == "child_created" and second["outcome"] == "child_created"
    assert len(children) == len(reservations) == 1
    assert children[0].goal_type == "outcome" and children[0].continuous_origin_key == run.cycle_key
    assert successor.id != run.id and successor.phase == "waiting_activation"
    assert run.status == "completed" and run.phase == "completed"


async def test_settled_late_child_recovers_current_health(db_session, test_project):
    goal, run = await started_continuous(db_session, test_project)
    continuous = OrchestrationContinuousService(OrchestrationService())
    due = datetime.fromisoformat(goal.continuous_state["next_due_at"])
    await continuous.claim_due_cycle(db_session, goal.id, due)
    await continuous.advance(db_session, goal, run, due)
    child = await db_session.scalar(select(OrchestrationGoal).where(
        OrchestrationGoal.parent_goal_id == goal.id,
    ))
    child.status = "cancelled"
    child_run = await OrchestrationService().get_run_for_goal(db_session, test_project.id, child.id)
    child_run.status = child_run.phase = "completed"
    child_run.started_at = due
    child_run.completed_at = due + timedelta(seconds=goal.continuous_policy["response_target_seconds"] + 1)
    assert await continuous.settle_terminal_children(db_session, goal, child_run.completed_at) == 1
    assert goal.continuous_state["health"] == "healthy"


async def test_automatic_stop_terminalizes_waiting_cycle_without_successor(db_session, test_project):
    goal, run = await started_continuous(db_session, test_project)
    policy = direct_policy(stop_condition={"mode": "max_cycles", "max_cycles": 1})
    await OrchestrationContinuousService(OrchestrationService()).update_policy(
        db_session, test_project.id, goal.id, OrchestrationContinuousPolicy.model_validate(policy), actor="human:test",
    )
    goal.continuous_state = {**goal.continuous_state, "cycles_completed": 1}
    due = datetime.fromisoformat(goal.continuous_state["next_due_at"])
    assert await OrchestrationContinuousService(OrchestrationService()).claim_due_cycle(db_session, goal.id, due) is None
    assert run.status == run.phase == "completed"
    assert goal.continuous_state["stopped_at"] is not None


async def test_claimed_cycle_uses_its_policy_snapshot_for_automatic_stop(db_session, test_project):
    goal, run = await started_continuous(db_session, test_project, enabled=False)
    continuous = OrchestrationContinuousService(OrchestrationService())
    due = datetime.fromisoformat(goal.continuous_state["next_due_at"])
    await continuous.claim_due_cycle(db_session, goal.id, due)
    await continuous.update_policy(db_session, test_project.id, goal.id, OrchestrationContinuousPolicy.model_validate(
        direct_policy(stop_condition={"mode": "deadline", "deadline": (due - timedelta(seconds=1)).isoformat()}),
    ), actor="human:test")
    assert (await continuous.advance(db_session, goal, run, due))["outcome"] == "no_action"


async def test_direct_child_authorization_and_event_include_contract_fields(db_session, test_project):
    goal, run = await started_continuous(db_session, test_project)
    continuous = OrchestrationContinuousService(OrchestrationService())
    due = datetime.fromisoformat(goal.continuous_state["next_due_at"])
    await continuous.claim_due_cycle(db_session, goal.id, due)
    await continuous.advance(db_session, goal, run, due)
    child = await db_session.scalar(select(OrchestrationGoal).where(OrchestrationGoal.parent_goal_id == goal.id))
    child_run = await OrchestrationService().get_run_for_goal(db_session, test_project.id, child.id)
    authorization = await db_session.scalar(select(OrchestrationAction).where(
        OrchestrationAction.run_id == child_run.id, OrchestrationAction.action_type == "authorize_execution",
    ))
    event = await db_session.scalar(select(EventLog).where(
        EventLog.event_type == "orchestration.continuous_child_released",
    ))
    assert authorization.request["action_type"] == "authorize_execution"
    assert event.payload["child_run_id"] == str(child_run.id)


async def test_snapshot_deadline_closes_authorized_cycle_once_as_stopped(db_session, test_project):
    goal, run = await started_continuous(db_session, test_project, enabled=False)
    due = datetime.fromisoformat(goal.continuous_state["next_due_at"])
    policy = direct_policy(stop_condition={"mode": "deadline", "deadline": (due + timedelta(hours=1)).isoformat()})
    continuous = OrchestrationContinuousService(OrchestrationService())
    await continuous.update_policy(db_session, test_project.id, goal.id, OrchestrationContinuousPolicy.model_validate(policy), actor="human:test")
    due = datetime.fromisoformat(goal.continuous_state["next_due_at"])
    await continuous.claim_due_cycle(db_session, goal.id, due)
    cycles_before = goal.continuous_state["cycles_completed"]
    assert (await continuous.advance(db_session, goal, run, due + timedelta(hours=2)))["outcome"] == "stopped"
    assert (await continuous.advance(db_session, goal, run, due + timedelta(hours=2)))["outcome"] == "stopped"
    assert await OrchestrationService().get_active_run_for_goal(db_session, test_project.id, goal.id) is None
    assert goal.continuous_state["cycles_completed"] == cycles_before + 1
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id,
        OrchestrationAction.action_type == "complete_cycle",
    )) == 1


async def test_max_cycles_closes_authorized_cycle_once_as_stopped(db_session, test_project):
    goal, run = await started_continuous(db_session, test_project, enabled=False)
    continuous = OrchestrationContinuousService(OrchestrationService())
    policy = direct_policy(stop_condition={"mode": "max_cycles", "max_cycles": 1})
    await continuous.update_policy(
        db_session, test_project.id, goal.id, OrchestrationContinuousPolicy.model_validate(policy), actor="human:test",
    )
    due = datetime.fromisoformat(goal.continuous_state["next_due_at"])
    await continuous.claim_due_cycle(db_session, goal.id, due)
    goal.continuous_state = {**goal.continuous_state, "cycles_completed": 1}

    assert (await continuous.advance(db_session, goal, run, due))["outcome"] == "stopped"
    assert goal.continuous_state["cycles_completed"] == 2
    assert await OrchestrationService().get_active_run_for_goal(db_session, test_project.id, goal.id) is None
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id,
        OrchestrationAction.action_type == "complete_cycle",
    )) == 1


async def test_manual_stop_closes_authorized_cycle_once_as_stopped(db_session, test_project):
    goal, run = await started_continuous(db_session, test_project, enabled=False)
    continuous = OrchestrationContinuousService(OrchestrationService())
    due = datetime.fromisoformat(goal.continuous_state["next_due_at"])
    await continuous.claim_due_cycle(db_session, goal.id, due)
    cycles_before = goal.continuous_state["cycles_completed"]

    await continuous.stop(db_session, test_project.id, goal.id, actor="human:test", reason="maintenance", now=due)
    assert (await continuous.advance(db_session, goal, run, due))["outcome"] == "stopped"

    assert run.status == run.phase == "completed"
    assert goal.continuous_state["cycles_completed"] == cycles_before + 1
    assert await OrchestrationService().get_active_run_for_goal(db_session, test_project.id, goal.id) is None
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id,
        OrchestrationAction.action_type == "complete_cycle",
    )) == 1


async def test_active_child_slo_uses_originating_cycle_policy_after_parent_policy_update(db_session, test_project):
    goal, run = await started_continuous(db_session, test_project)
    continuous = OrchestrationContinuousService(OrchestrationService())
    due = datetime.fromisoformat(goal.continuous_state["next_due_at"])
    await continuous.update_policy(
        db_session, test_project.id, goal.id,
        OrchestrationContinuousPolicy.model_validate(direct_policy(response_target_seconds=1)), actor="human:test",
    )
    await continuous.claim_due_cycle(db_session, goal.id, due)
    await continuous.advance(db_session, goal, run, due)
    child = await db_session.scalar(select(OrchestrationGoal).where(OrchestrationGoal.parent_goal_id == goal.id))
    child_run = await OrchestrationService().get_run_for_goal(db_session, test_project.id, child.id)
    child_run.started_at = due
    await continuous.update_policy(
        db_session, test_project.id, goal.id,
        OrchestrationContinuousPolicy.model_validate(direct_policy(response_target_seconds=900)), actor="human:test",
    )

    health = await continuous.derive_health(db_session, goal, due + timedelta(seconds=2))

    assert health["health"] == "degraded"
    assert health["reason"] == "response_target_missed"


async def test_budget_wait_rolls_back_release_action_and_overflow_replays_cleanly(db_session, test_project):
    policy = direct_policy(
        max_backlog=1,
        per_case_budget={"max_tokens": "50", "max_turns": "1", "max_hours": "0.25"},
        rolling_budget={"window_seconds": 3600, "limits": {"max_tokens": "50", "max_turns": "1", "max_hours": "0.25"}},
    )
    goal, run = await continuous_ready(db_session, test_project)
    continuous = OrchestrationContinuousService(OrchestrationService())
    await continuous.update_policy(
        db_session, test_project.id, goal.id, OrchestrationContinuousPolicy.model_validate(policy), actor="human:test",
    )
    await OrchestrationService().start_run(db_session, test_project.id, goal.id, actor="human:test")
    due = datetime.fromisoformat(goal.continuous_state["next_due_at"])
    holding_child = OrchestrationGoal(
        project_id=test_project.id, objective="Holding child", original_request="Holding child",
    )
    db_session.add(holding_child)
    await db_session.flush()
    db_session.add(OrchestrationBudgetReservation(
        parent_goal_id=goal.id, roadmap_item_id=None, child_goal_id=holding_child.id,
        continuous_origin_key="holds-window", allocation=policy["per_case_budget"],
    ))
    await db_session.flush()

    await continuous.claim_due_cycle(db_session, goal.id, due)
    assert (await continuous.advance(db_session, goal, run, due))["outcome"] == "budget_wait"
    assert (await continuous.advance(db_session, goal, run, due))["outcome"] == "budget_wait"
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == "release_continuous_child",
    )) == 0
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationGoal).where(
        OrchestrationGoal.parent_goal_id == goal.id,
    )) == 0
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.parent_goal_id == goal.id,
    )) == 1
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id,
        OrchestrationAction.action_type == "complete_cycle",
        OrchestrationAction.status == "completed",
    )) == 1
    assert len(goal.continuous_state["pending_candidates"]) == 1

    successor = await OrchestrationService().get_active_run_for_goal(db_session, test_project.id, goal.id)
    next_due = datetime.fromisoformat(goal.continuous_state["next_due_at"])
    assert await continuous.claim_due_cycle(db_session, goal.id, next_due) is None
    assert successor.phase == "waiting_activation"
    assert len(goal.continuous_state["pending_candidates"]) == 1
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
        OrchestrationAction.action_type == "release_continuous_child",
    )) == 0


async def test_cancel_continuous_settles_released_child_reservation(db_session, test_project):
    goal, run = await started_continuous(db_session, test_project)
    continuous = OrchestrationContinuousService(OrchestrationService())
    due = datetime.fromisoformat(goal.continuous_state["next_due_at"])
    await continuous.claim_due_cycle(db_session, goal.id, due)
    await continuous.advance(db_session, goal, run, due)
    child = await db_session.scalar(select(OrchestrationGoal).where(OrchestrationGoal.parent_goal_id == goal.id))
    reservation = await db_session.scalar(select(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.child_goal_id == child.id,
    ))

    await OrchestrationService().cancel_goal(db_session, test_project.id, goal.id, cancelled_by="human:test")

    assert child.status == "cancelled"
    assert reservation.status == "settled"
    assert reservation.settlement_reason == "cancelled"


async def test_started_continuous_goal_cannot_reset_execution_lineage(db_session, test_project):
    goal, _ = await started_continuous(db_session, test_project)

    with pytest.raises(HTTPException, match="immutable execution lineage") as exc:
        await OrchestrationService().reset_goal(db_session, test_project.id, goal.id)

    assert exc.value.status_code == 409


async def test_reset_fresh_continuous_goal_preserves_policy_and_clears_state(db_session, test_project):
    policy = {"version": 1, **direct_policy()}
    goal = OrchestrationGoal(
        project_id=test_project.id,
        objective="Fresh continuous",
        original_request="Fresh continuous",
        goal_type="continuous",
        continuous_policy=deepcopy(policy),
        continuous_state={"health": "degraded", "pending_slots": ["stale"]},
    )
    db_session.add(goal)
    await db_session.flush()
    db_session.add(OrchestrationRun(goal_id=goal.id))
    await db_session.flush()

    reset_goal, _ = await OrchestrationService().reset_goal(db_session, test_project.id, goal.id)

    assert reset_goal.continuous_policy == policy
    assert reset_goal.continuous_state == {}


async def test_child_run_blocker_needs_attention_then_recovers_when_settled(db_session, test_project):
    goal, _ = await started_continuous(db_session, test_project)
    child = OrchestrationGoal(
        project_id=test_project.id,
        objective="Active child",
        original_request="Active child",
        parent_goal_id=goal.id,
        continuous_origin_key="active-child",
        parent_contract_snapshot={},
        goal_delta={},
    )
    db_session.add(child)
    await db_session.flush()
    child_run = OrchestrationRun(
        goal_id=child.id,
        active_blockers=[{"kind": "child_runtime_blocker"}],
    )
    db_session.add(child_run)
    await db_session.flush()
    continuous = OrchestrationContinuousService(OrchestrationService())

    assert (await continuous.derive_health(db_session, goal))["health"] == "needs_attention"
    assert goal.continuous_state["health_reason"] == "child_runtime_blocker"

    child_run.active_blockers = []
    assert (await continuous.derive_health(db_session, goal))["health"] == "healthy"

    child.status = child_run.status = child_run.phase = "completed"
    assert (await continuous.derive_health(db_session, goal))["health"] == "healthy"


async def test_active_child_blocked_run_is_not_hidden_by_completed_history(db_session, test_project):
    goal, _ = await started_continuous(db_session, test_project)
    child = OrchestrationGoal(
        project_id=test_project.id,
        objective="Active child",
        original_request="Active child",
        parent_goal_id=goal.id,
        continuous_origin_key="active-child",
        parent_contract_snapshot={},
        goal_delta={},
    )
    db_session.add(child)
    await db_session.flush()
    db_session.add_all([
        OrchestrationRun(goal_id=child.id, status="completed", phase="completed"),
        OrchestrationRun(
            goal_id=child.id,
            status="blocked",
            active_blockers=[{"kind": "child_runtime_blocker"}],
        ),
    ])
    await db_session.flush()

    result = await OrchestrationContinuousService(OrchestrationService()).derive_health(db_session, goal)

    assert result["health"] == "needs_attention"
    assert result["reason"] == "child_runtime_blocker"


async def test_successful_settlement_clears_scoped_budget_blocker(db_session, test_project):
    goal, run = await started_continuous(db_session, test_project)
    continuous = OrchestrationContinuousService(OrchestrationService())
    due = datetime.fromisoformat(goal.continuous_state["next_due_at"])
    await continuous.claim_due_cycle(db_session, goal.id, due)
    await continuous.advance(db_session, goal, run, due)
    child = await db_session.scalar(select(OrchestrationGoal).where(OrchestrationGoal.parent_goal_id == goal.id))
    reservation = await db_session.scalar(select(OrchestrationBudgetReservation).where(
        OrchestrationBudgetReservation.child_goal_id == child.id,
    ))
    child.status = "cancelled"
    child_run = await OrchestrationService().get_run_for_goal(db_session, test_project.id, child.id)
    child_run.status = child_run.phase = "completed"
    successor = await OrchestrationService().get_active_run_for_goal(db_session, test_project.id, goal.id)
    successor.active_blockers = [{"kind": "budget_integrity", "origin_key": reservation.continuous_origin_key}]
    await continuous.settle_terminal_children(db_session, goal, due)
    assert successor.active_blockers == []


async def test_successful_budget_measurement_clears_goal_scoped_blocker(db_session, test_project, monkeypatch):
    goal, run = await started_continuous(db_session, test_project)
    continuous = OrchestrationContinuousService(OrchestrationService())
    original = OrchestrationBudgetService.continuous_available

    async def missing_measurement(*_args):
        raise BudgetMeasurementError("max_tokens", uuid.uuid4())

    monkeypatch.setattr(OrchestrationBudgetService, "continuous_available", missing_measurement)
    await continuous.derive_health(db_session, goal)
    assert any(blocker.get("scope") == f"continuous:{goal.id}" for blocker in run.active_blockers)

    monkeypatch.setattr(OrchestrationBudgetService, "continuous_available", original)
    await continuous.derive_health(db_session, goal)
    assert not any(blocker.get("scope") == f"continuous:{goal.id}" for blocker in run.active_blockers)


async def test_claimed_cycle_without_snapshot_needs_attention_not_mutable_stop(db_session, test_project):
    goal, run = await started_continuous(db_session, test_project, enabled=False)
    continuous = OrchestrationContinuousService(OrchestrationService())
    due = datetime.fromisoformat(goal.continuous_state["next_due_at"])
    await continuous.claim_due_cycle(db_session, goal.id, due)
    run.plan_state = {}
    goal.continuous_policy = {"version": 2, **direct_policy(
        stop_condition={"mode": "deadline", "deadline": (due - timedelta(seconds=1)).isoformat()},
    )}
    assert await continuous.claim_due_cycle(db_session, goal.id, due) is None
    assert run.status == "running"
    assert any(blocker["kind"] == "continuous_integrity" for blocker in run.active_blockers)


async def test_capacity_leaves_due_slot_unclaimed_until_child_finishes(db_session, test_project):
    goal, run = await continuous_ready(db_session, test_project)
    continuous = OrchestrationContinuousService(OrchestrationService())
    await continuous.update_policy(
        db_session,
        test_project.id,
        goal.id,
        OrchestrationContinuousPolicy.model_validate(direct_policy(max_active_cases=1)),
        actor="human:test",
    )
    await OrchestrationService().start_run(db_session, test_project.id, goal.id, actor="human:test")
    due = datetime.fromisoformat(goal.continuous_state["next_due_at"])
    await continuous.claim_due_cycle(db_session, goal.id, due)
    await continuous.advance(db_session, goal, run, due)
    successor = await OrchestrationService().get_active_run_for_goal(db_session, test_project.id, goal.id)
    next_due = goal.continuous_state["next_due_at"]

    assert await continuous.claim_due_cycle(db_session, goal.id, datetime.fromisoformat(next_due)) is None
    assert successor.phase == "waiting_activation"
    assert goal.continuous_state["next_due_at"] == next_due


async def test_pause_stops_claims_without_cancelling_existing_child(db_session, test_project):
    goal, run = await started_continuous(db_session, test_project)
    continuous = OrchestrationContinuousService(OrchestrationService())
    due = datetime.fromisoformat(goal.continuous_state["next_due_at"])
    await continuous.claim_due_cycle(db_session, goal.id, due)
    await continuous.advance(db_session, goal, run, due)
    child = await db_session.scalar(select(OrchestrationGoal).where(OrchestrationGoal.parent_goal_id == goal.id))
    successor = await OrchestrationService().get_active_run_for_goal(db_session, test_project.id, goal.id)
    next_due = datetime.fromisoformat(goal.continuous_state["next_due_at"])

    await OrchestrationService().pause_goal(db_session, test_project.id, goal.id)

    assert await continuous.claim_due_cycle(db_session, goal.id, next_due) is None
    assert child.status == "active"
    assert successor.status == "paused"


@pytest.mark.parametrize("reauthorized", [False, True])
async def test_continuous_successor_keeps_authorized_fingerprint(db_session, test_project, reauthorized):
    goal, run = await started_continuous(db_session, test_project)
    old = run.goal_contract_fingerprint
    assert old == compute_goal_contract_fingerprint(goal)
    goal.objective = "changed objective"
    new = compute_goal_contract_fingerprint(goal)
    assert new != old
    if reauthorized:
        db_session.add(OrchestrationAction(
            run_id=run.id, idempotency_key=reauthorize_execution_key(run.id, new),
            action_type="reauthorize_execution", status="completed",
        ))
        await db_session.flush()
    continuous = OrchestrationContinuousService(OrchestrationService())
    await continuous._complete_cycle(db_session, goal, run, "no_action", datetime.now(timezone.utc))
    successor = await db_session.get(OrchestrationRun, uuid.uuid5(uuid.NAMESPACE_URL, f"rally:continuous-successor:{run.id}"))
    assert successor.goal_contract_fingerprint == (new if reauthorized else old)


# --- discovery follows the attempt an approved fix started ---------------------------

_VALID_BATCH = '{"schema_version":1,"candidates":[]}'


def _complete(session, output):
    session.status = "completed"
    session.output = output
    session.metadata_ = {**(session.metadata_ or {}), "token_count_in": 0, "token_count_out": 0,
                         "token_usage_complete": True, "_roadmap_elapsed_seconds": 0}


async def _lineage_proposal(db, run, blocker_code):
    """A recovery proposal in the lineage of the discovery trigger (what approve_fix acts on)."""
    trigger = (await _discovery_triggers(db, run, blocker_code))[0]
    pid = uuid.uuid4()
    proposal = OrchestrationAction(
        run_id=run.id, idempotency_key=f"recovery:{pid}:proposal", action_type="recovery_proposal",
        status="reserved", request={"proposal_id": str(pid)},
        dispatch_contract={"trigger_action_id": str(trigger.id), "lineage_id": str(uuid.uuid4()), "attempt": 1},
    )
    db.add(proposal)
    await db.flush()
    trigger.status, trigger.target_type, trigger.target_id = "completed", "orchestration_action", proposal.id  # diagnosed
    await db.flush()
    return pid, proposal


async def _failed_source(db, task, session):
    task.status = session.status = "failed"
    session.metadata_ = {**session.metadata_, "token_count_in": 0, "token_count_out": 0,
                         "token_usage_complete": True, "_roadmap_elapsed_seconds": 0,
                         "attempt": {"effect_state": "started"}}
    await db.flush()


@pytest.mark.parametrize("tool", ["retry_task", "reassign_task"])
async def test_discovery_consumes_the_session_an_approved_fix_started(
    db_session, test_project, test_agent, monkeypatch, tool,
):
    """Fix round 3 (R2): any lineage effect that starts a new session of the source task is followed and consumed."""
    goal, run, task, session, continuous = await dispatched_discovery_source(
        db_session, test_project, test_agent, monkeypatch,
    )
    await _failed_source(db_session, task, session)
    assert (await continuous.advance(db_session, goal, run))["step"] == "waiting_discovery_recovery"
    pid, _proposal = await _lineage_proposal(db_session, run, "continuous_discovery_source_failed")

    service = continuous.orchestration
    key = f"recovery:{pid}:apply:call-1:effect"
    if tool == "retry_task":
        effect = await service.execute_retry_task_action(
            db_session, run.id, {"action_type": "retry_task", "task_id": str(task.id), "timeout": 300,
                                 "max_tokens": 100}, key)
    else:
        effect = await service.execute_reassign_task_action(
            db_session, run.id, {"action_type": "reassign_task", "task_id": str(task.id),
                                 "agent_id": str(test_agent.id)}, key)
    assert effect.status == "completed" and effect.target_id != session.id, effect.error
    attempt = await db_session.get(Session, effect.target_id)
    task.status = "done"
    _complete(attempt, _VALID_BATCH)
    await db_session.flush()

    await continuous.advance(db_session, goal, run)

    discovery = run.plan_state["discovery"]
    assert discovery["active_session_id"] == str(attempt.id)
    assert discovery["accepted_batch"] == {"schema_version": 1, "candidates": []}
    assert discovery["accepted_session_id"] == str(attempt.id)


async def test_discovery_dispatches_a_fresh_source_attempt_after_a_delegated_remediation(
    db_session, test_project, test_agent, monkeypatch,
):
    """Fix round 3 (R1): a delegated remediation (no parent_task_id) is never parsed; once its sealed retry ran,
    discovery dispatches one fresh source attempt and consumes that."""
    goal, run, task, session, continuous = await dispatched_discovery_source(
        db_session, test_project, test_agent, monkeypatch,
    )
    task.status = "done"
    _complete(session, "not json")
    assert (await continuous.advance(db_session, goal, run))["step"] == "waiting_discovery_recovery"
    pid, proposal = await _lineage_proposal(db_session, run, "continuous_discovery_output_invalid")
    remediation = await continuous.orchestration.execute_create_delegation_task_action(
        db_session, run.id, {
            "action_type": "create_delegation_task", "agent_id": str(test_agent.id), "work_function": "implementation",
            "scope": "Fix the discovery source access.", "deliverable": "Source access fixed.",
            "budget": {"max_tokens": "100"},
        }, f"recovery:{pid}:apply:call-1:effect", skip_baseline_gate=True,
    )
    assert remediation.status == "completed", remediation.error
    # Applied but not yet verified/retried: still waiting, nothing dispatched.
    assert (await continuous.advance(db_session, goal, run))["step"] == "waiting_discovery_recovery"
    assert run.plan_state["discovery"]["active_task_id"] == str(task.id)

    retry = await continuous.orchestration.reserve_action(
        db_session, run.id, f"recovery:{pid}:retry:1", "recovery_retry", {"proposal_action_id": str(proposal.id)})
    result = await continuous.advance(db_session, goal, run)

    assert result["step"] == "discovery_running"
    fresh_id = result["task_id"]
    assert fresh_id not in {str(task.id), str(remediation.target_id)}
    discovery = run.plan_state["discovery"]
    assert discovery["active_task_id"] == discovery["repair_task_id"] == fresh_id
    assert discovery["source_task_id"] == str(task.id)
    assert discovery["recovery_attempt_action_id"] == str(retry.id)
    # Replay never dispatches a second attempt.
    assert (await continuous.advance(db_session, goal, run))["step"] == "discovery_running"
    assert run.plan_state["discovery"]["active_task_id"] == fresh_id

    fresh = await db_session.get(Task, uuid.UUID(fresh_id))
    fresh_session = await db_session.get(Session, uuid.UUID(run.plan_state["discovery"]["active_session_id"]))
    fresh.status = "done"
    _complete(fresh_session, _VALID_BATCH)
    await db_session.flush()
    await continuous.advance(db_session, goal, run)
    assert run.plan_state["discovery"]["accepted_task_id"] == fresh_id


@pytest.mark.parametrize("lineage_open", [True, False])
async def test_recovery_attempt_with_invalid_output_fails_into_its_lineage_or_needs_attention(
    db_session, test_project, test_agent, monkeypatch, lineage_open,
):
    """Fix round 3 (R3): a recovery-produced attempt that ends done-with-invalid-output never waits silently."""
    goal, run, task, session, continuous = await dispatched_discovery_source(
        db_session, test_project, test_agent, monkeypatch,
    )
    await _failed_source(db_session, task, session)
    assert (await continuous.advance(db_session, goal, run))["step"] == "waiting_discovery_recovery"
    pid, proposal = await _lineage_proposal(db_session, run, "continuous_discovery_source_failed")
    if not lineage_open:
        proposal.status = "completed"  # lineage already resolved: a new bad output is not its evidence
    effect = await continuous.orchestration.execute_retry_task_action(
        db_session, run.id, {"action_type": "retry_task", "task_id": str(task.id), "timeout": 300, "max_tokens": 100},
        f"recovery:{pid}:apply:call-1:effect",
    )
    task.status = "done"
    _complete(await db_session.get(Session, effect.target_id), "not json")
    await db_session.flush()

    result = await continuous.advance(db_session, goal, run)

    if lineage_open:
        assert result["step"] == "waiting_discovery_recovery"
        assert proposal.status == "failed"
        follow_up = await continuous.orchestration._existing_action_for_key(  # pylint: disable=protected-access
            db_session, run.id, f"recovery:trigger:{proposal.dispatch_contract['trigger_action_id']}:attempt:2")
        assert follow_up is not None and follow_up.status == "reserved"
        assert (await continuous.advance(db_session, goal, run))["step"] == "waiting_discovery_recovery"
        # The follow-up closes without a newer proposal: attention, never a silent wait.
        follow_up.status = "completed"
        follow_up.target_id = follow_up.id
        await db_session.flush()
        assert (await continuous.advance(db_session, goal, run))["step"] == "needs_attention"
    else:
        assert result["step"] == "needs_attention"
        assert any(b["kind"] == "continuous_discovery_output_invalid" for b in run.active_blockers)
    assert len(await _discovery_triggers(db_session, run, "continuous_discovery_output_invalid")) == 0


async def test_non_canonical_terminal_session_stays_in_attention_without_flip_flop(
    db_session, test_project, test_agent, monkeypatch,
):
    goal, run, task, session, continuous = await dispatched_discovery_source(
        db_session, test_project, test_agent, monkeypatch,
    )
    task.status = "done"
    _complete(session, _VALID_BATCH)
    run.plan_state = {**run.plan_state, "discovery": {**run.plan_state["discovery"],
                                                      "active_session_id": str(uuid.uuid4())}}
    await db_session.flush()

    for _ in range(3):
        assert (await continuous.advance(db_session, goal, run))["step"] == "needs_attention"
        assert any(b["kind"] == "continuous_discovery_source_execution_failed" for b in run.active_blockers)


async def test_discovery_execution_blocker_reactivates_and_clears(db_session, test_project, test_agent, monkeypatch):
    """advance()'s blocker-clearing/reactivation branch: a recoverable execution blocker is cleared once the
    assignment is valid and the cycle reservation reactivates; the source attempt then continues."""
    goal, run, task, _session, continuous = await dispatched_discovery_source(
        db_session, test_project, test_agent, monkeypatch,
    )
    real_task_id = run.plan_state["discovery"]["active_task_id"]
    run.plan_state = {**run.plan_state, "discovery": {**run.plan_state["discovery"], "active_task_id": str(uuid.uuid4())}}
    await db_session.flush()
    assert (await continuous.advance(db_session, goal, run))["step"] == "needs_attention"
    assert any(b["kind"] == "continuous_discovery_source_execution_failed" for b in run.active_blockers)

    run.plan_state = {**run.plan_state, "discovery": {**run.plan_state["discovery"], "active_task_id": real_task_id}}
    await db_session.flush()
    assert (await continuous.advance(db_session, goal, run))["step"] == "discovery_running"
    assert not any(b["kind"] == "continuous_discovery_source_execution_failed" for b in run.active_blockers)


# --- discovery resolves its lineage; every fresh attempt stays budget-accounted ----


async def _active_recovery_warnings(db, run):
    from huddleroom.models.orchestration_process import OrchestrationWarning

    return (await db.scalars(select(OrchestrationWarning).where(
        OrchestrationWarning.run_id == run.id, OrchestrationWarning.active.is_(True),
        OrchestrationWarning.warning_type == "recovery_blocked"))).all()


@pytest.mark.parametrize("verification_pending", [False, True])
async def test_discovery_consume_resolves_an_approved_retry_lineage(
    owned_recovery_setup, monkeypatch, verification_pending,
):
    """I1: a real approved retry_task lineage; discovery's valid consume drives verify + retry:1 and resolves it."""
    from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
    from huddleroom.services.orchestration_blocker_recovery import OrchestrationBlockerRecoveryService
    from tests.test_orchestration_recovery_activation import _Scripted, _retry_draft
    from tests.test_orchestration_roster_mutation import make_agent

    db_session, test_project, test_user = owned_recovery_setup
    test_agent = make_agent(name=f"discovery-retry-{uuid.uuid4()}", capabilities=["implementation"])
    db_session.add(test_agent)
    await db_session.flush()
    goal, run, task, session, continuous = await dispatched_discovery_source(
        db_session, test_project, test_agent, monkeypatch,
    )
    await _failed_source(db_session, task, session)
    assert (await continuous.advance(db_session, goal, run))["step"] == "waiting_discovery_recovery"
    (trigger,) = await _discovery_triggers(db_session, run, "continuous_discovery_source_failed")
    svc = OrchestrationBlockerRecoveryService(analyzer=_Scripted(_retry_draft(task.id)))
    proposal = await svc.diagnose(db_session, trigger.id)
    assert proposal.action_type == "recovery_proposal", proposal
    decision = (await db_session.scalars(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.related_action_id == proposal.id))).one()
    await svc.reserve_answer(db_session, decision, selected_option="approve_fix", reason=None, user_id=test_user.id)
    await svc.execute_next_call(db_session, proposal)  # apply only: the retry_task effect starts a new session
    assert await _active_recovery_warnings(db_session, run)

    effect = next(row for row in (await db_session.scalars(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == "retry_task"))).all()
        if row.idempotency_key.endswith(":effect"))
    attempt = await db_session.get(Session, effect.target_id)
    task.status = "done"
    _complete(attempt, _VALID_BATCH)
    await db_session.flush()

    if verification_pending:
        # Fix round 5: verification has not passed yet -> the accepted batch holds; nothing fans out.
        async def pending(self, db, proposal_action):
            return ()

        with monkeypatch.context() as patch:
            patch.setattr(OrchestrationBlockerRecoveryService, "verify_proposal", pending)
            assert (await continuous.advance(db_session, goal, run))["step"] == "waiting_discovery_recovery"
            assert (await continuous.advance(db_session, goal, run))["step"] == "waiting_discovery_recovery"
        assert run.plan_state["discovery"]["accepted_batch"] is not None and run.status != "completed"
        await db_session.refresh(proposal)
        assert proposal.status == "reserved"

    result = await continuous.advance(db_session, goal, run)

    assert result == {"step": "complete_cycle", "outcome": "no_action"}
    assert run.plan_state["discovery"]["accepted_session_id"] == str(attempt.id)
    await db_session.refresh(proposal)
    assert proposal.status == "completed"
    assert await svc.resume(db_session, proposal.id) == "resolved"
    assert await _active_recovery_warnings(db_session, run) == []


async def test_discovery_fresh_attempt_consume_resolves_the_delegated_remediation_lineage(
    db_session, test_project, test_agent, monkeypatch,
):
    """I1 (delegate path): after the fresh source attempt's valid consume, discovery drives the lineage's
    lifecycle (resume) and completes its reserved decision_loop retry. The lifecycle itself is pinned end to end by
    test_complete_retry_on_observed_progress_resolves_the_lineage (recovery activation tests)."""
    from huddleroom.services.orchestration_blocker_recovery import OrchestrationBlockerRecoveryService

    calls = []

    async def resume(self, db, proposal_action_id):
        calls.append(("resume", proposal_action_id))

    async def complete(self, db, proposal_action):
        calls.append(("complete", proposal_action.id))
        proposal_action.status = "completed"
        await db.flush()
        return True

    monkeypatch.setattr(OrchestrationBlockerRecoveryService, "resume", resume)
    monkeypatch.setattr(OrchestrationBlockerRecoveryService, "complete_retry_on_observed_progress", complete)
    goal, run, task, session, continuous = await dispatched_discovery_source(
        db_session, test_project, test_agent, monkeypatch,
    )
    task.status = "done"
    _complete(session, "not json")
    assert (await continuous.advance(db_session, goal, run))["step"] == "waiting_discovery_recovery"
    pid, proposal = await _lineage_proposal(db_session, run, "continuous_discovery_output_invalid")
    await continuous.orchestration.reserve_action(
        db_session, run.id, f"recovery:{pid}:retry:1", "recovery_retry", {"proposal_action_id": str(proposal.id)})
    fresh_id = (await continuous.advance(db_session, goal, run))["task_id"]
    assert calls == []
    fresh = await db_session.get(Task, uuid.UUID(fresh_id))
    fresh.status = "done"
    _complete(await db_session.get(Session, uuid.UUID(run.plan_state["discovery"]["active_session_id"])), _VALID_BATCH)
    await db_session.flush()

    await continuous.advance(db_session, goal, run)

    assert run.plan_state["discovery"]["accepted_task_id"] == fresh_id
    assert calls == [("resume", proposal.id), ("complete", proposal.id)]
    assert proposal.status == "completed"


async def test_two_fresh_attempts_both_count_against_the_discovery_reservation(
    db_session, test_project, test_agent, monkeypatch,
):
    """I2: a second fresh source attempt never drops the first attempt's spend from the cycle's accounting."""
    goal, run, task, session, continuous = await dispatched_discovery_source(
        db_session, test_project, test_agent, monkeypatch,
    )
    async def fresh_attempt(previous_task, previous_session, pid):
        previous_task.status = "done"
        _complete(previous_session, "not json")
        run.plan_state = {**run.plan_state, "discovery": {**run.plan_state["discovery"], "active_session_id": None}}
        await db_session.flush()
        return await continuous.dispatch_discovery_source(db_session, goal, run, datetime.now(timezone.utc),
                                                          attempt=pid)

    first = await fresh_attempt(task, session, "p1")
    assert first is not None, run.active_blockers
    first_session = await db_session.get(Session, uuid.UUID(run.plan_state["discovery"]["active_session_id"]))
    # The per-cycle cap is 2 turns: source + first attempt used both, so the second attempt is refused. Before the
    # fix the first attempt's turn dropped out of the accounting and the second attempt ran past the cap.
    assert await fresh_attempt(first, first_session, "p2") is None
    assert any(b.get("reason") == "Execution budget is exhausted" for b in run.active_blockers)

    discovery = run.plan_state["discovery"]
    second_id = uuid.UUID(discovery["repair_task_id"])
    assert discovery["repair_task_ids"] == [str(first.id), str(second_id)]
    assert discovery["source_task_id"] == str(task.id)
    assert OrchestrationBudgetService._discovery_task_ids(run) == {task.id, first.id, second_id}  # pylint: disable=protected-access
    # Legacy state (a single repair_task_id) is still read.
    run.plan_state = {**run.plan_state, "discovery": {"source_task_id": str(task.id), "repair_task_id": str(first.id)}}
    assert OrchestrationBudgetService._discovery_task_ids(run) == {task.id, first.id}  # pylint: disable=protected-access


async def test_discovery_lineage_error_is_logged_and_fan_out_proceeds(
    db_session, test_project, test_agent, monkeypatch, caplog,
):
    """Fix round 5: a lineage error at the consume site never blocks fan-out or loops every tick."""
    from huddleroom.services.orchestration_blocker_recovery import OrchestrationBlockerRecoveryService

    calls = []

    async def broken(self, db, proposal_action_id):
        calls.append(proposal_action_id)
        raise HTTPException(status_code=409, detail="baseline unavailable")

    monkeypatch.setattr(OrchestrationBlockerRecoveryService, "resume", broken)
    goal, run, task, session, continuous = await dispatched_discovery_source(
        db_session, test_project, test_agent, monkeypatch,
    )
    await _failed_source(db_session, task, session)
    assert (await continuous.advance(db_session, goal, run))["step"] == "waiting_discovery_recovery"
    pid, proposal = await _lineage_proposal(db_session, run, "continuous_discovery_source_failed")
    effect = await continuous.orchestration.execute_retry_task_action(
        db_session, run.id, {"action_type": "retry_task", "task_id": str(task.id), "timeout": 300, "max_tokens": 100},
        f"recovery:{pid}:apply:call-1:effect",
    )
    task.status = "done"
    _complete(await db_session.get(Session, effect.target_id), _VALID_BATCH)
    await db_session.flush()

    with caplog.at_level("WARNING"):
        result = await continuous.advance(db_session, goal, run)

    assert result == {"step": "complete_cycle", "outcome": "no_action"}
    assert "discovery lineage resolve skipped" in caplog.text
    assert calls == [proposal.id]
    assert (await continuous.advance(db_session, goal, run))["step"] == "complete_cycle"
    assert calls == [proposal.id]  # the completed cycle never re-enters the lineage


@pytest.mark.usefixtures("safe_goal_analysis", "safe_agent_definition_review", "safe_effectiveness_review_continue")
async def test_approved_team_configuration_fix_on_a_continuous_run_applies_within_subsequent_ticks(
    sessions, tmp_path, monkeypatch,
):
    """F5: apply is deferred under the goal lock (the continuous batch hold); the tick-level unlocked
    _drive_recovery pass must still run it for a continuous goal."""
    from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGoal
    from huddleroom.services.orchestration_blocker_recovery import OrchestrationBlockerRecoveryService
    from tests.test_orchestration_recovery_configuration import approved_flow, arm_semantic_guard, seed_spr180

    seeded = await seed_spr180(sessions, tmp_path)
    async with sessions() as s:
        goal = await s.get(OrchestrationGoal, seeded.goal_id)
        goal.goal_type, goal.planning_mode = "continuous", "continuous_policy"
        goal.continuous_policy = direct_policy()
        await s.commit()
    flow = await approved_flow(sessions, seeded)
    arm_semantic_guard(monkeypatch)
    async with sessions() as s:  # the continuous batch hold: resume under the goal lock defers the apply
        async with OrchestrationService()._lock_goal_for_baseline_transition(s, seeded.goal_id):
            await OrchestrationBlockerRecoveryService().resume(s, flow.proposal_action_id)
    async with sessions() as s:
        assert (await s.get(OrchestrationAction, flow.wrapper_id)).status == "reserved"

    for _ in range(3):
        async with sessions() as s:
            try:
                await OrchestrationService().tick(s, seeded.run_id)
            except Exception:  # noqa: BLE001 -- only the recovery pass matters here
                pass
        async with sessions() as s:
            if (await s.get(OrchestrationAction, flow.wrapper_id, populate_existing=True)).status == "completed":
                return
    raise AssertionError("approved team_configuration fix never applied across continuous ticks")
