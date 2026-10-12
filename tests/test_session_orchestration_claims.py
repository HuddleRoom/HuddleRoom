import uuid

import pytest

from fastapi import HTTPException

from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGoal, OrchestrationRun
from huddleroom.models.task import Task
from huddleroom.schemas.session import SessionCreate
from huddleroom.services.session_service import SessionService


async def _root_outcome(db, project, agent, caps, *, lineage=True):
    from huddleroom.services.orchestration_service import OrchestrationService

    goal = OrchestrationGoal(project_id=project.id, objective="Deliver", budget={"caps": caps})
    db.add(goal)
    await db.flush()
    run = OrchestrationRun(goal_id=goal.id, status="running", phase="authorized")
    db.add(run)
    await db.flush()
    task = Task(project_id=project.id, title="Deliver", assigned_to=agent.id,
                metadata_={"orchestration": {"run_id": str(run.id)}})
    db.add(task)
    await db.flush()
    db.add(OrchestrationAction(
        run_id=run.id, idempotency_key=OrchestrationService()._authorize_execution_key(run.id),
        action_type="authorize_execution", status="completed",
    ))
    action = OrchestrationAction(
        run_id=run.id, idempotency_key="deliver", action_type="create_delegation_task", status="completed",
        target_type="task", target_id=task.id,
    ) if lineage else None
    if action is not None:
        db.add(action)
    await db.flush()
    if action is not None:
        task.metadata_ = {"orchestration": {"run_id": str(run.id), "action_id": str(action.id)}}
        await db.flush()
    return goal, run, task, action


@pytest.mark.asyncio
async def test_root_outcome_claim_clamps_remaining_and_action_allocation(
    db_session, test_project, test_agent, monkeypatch,
):
    from huddleroom.models.orchestration import OrchestrationBudgetReservation
    from huddleroom.models.session import Session
    from huddleroom.services.orchestration_budget_service import OrchestrationBudgetService

    goal, run, task, action = await _root_outcome(
        db_session, test_project, test_agent, {"max_tokens": "100", "max_hours": "1"},
    )
    action.budget_ledger = {
        "allocation": {"max_tokens": "40", "max_hours": "0.75"}, "reserved": {},
        "committed": {"max_tokens": "40", "max_hours": "0.75"}, "consumed": {},
    }
    for title, status, metadata in [
        ("spent", "completed", {"token_count_in": 10, "token_count_out": 10, "_roadmap_elapsed_seconds": 900}),
        ("live", "running", {"_run_config": {"max_tokens": 10, "timeout": 450}}),
    ]:
        other = Task(project_id=test_project.id, title=title)
        db_session.add(other)
        await db_session.flush()
        db_session.add(OrchestrationAction(
            run_id=run.id, idempotency_key=title, action_type="create_delegation_task", status="completed",
            target_type="task", target_id=other.id,
        ))
        db_session.add(Session(project_id=test_project.id, task_id=other.id, agent_id=test_agent.id,
                               adapter_type="api", status=status, metadata_=metadata))
    child = OrchestrationGoal(project_id=test_project.id, objective="Retained child")
    db_session.add(child)
    await db_session.flush()
    db_session.add(OrchestrationBudgetReservation(
        parent_goal_id=goal.id, child_goal_id=child.id, continuous_origin_key="retained",
        allocation={"max_tokens": "20", "max_hours": "0.125"},
    ))
    await db_session.flush()
    assert (await OrchestrationBudgetService().remaining(db_session, goal))["remaining"] == {
        "max_tokens": "50", "max_hours": "0.500",
    }
    # Editable metadata cannot remove durable ownership.
    task.metadata_ = {"orchestration": {"action_id": str(action.id)}}
    run.active_blockers = [{"scope": f"claim:{goal.id}"}, {"scope": "claim:unrelated"}]
    scheduled = []
    monkeypatch.setattr(SessionService, "_schedule_dispatch_after_commit", lambda *args: scheduled.append(args[2]))
    session = await SessionService().create(db_session, SessionCreate(
        agent_id=test_agent.id, project_id=test_project.id, task_id=task.id,
    ))
    assert session.metadata_["_run_config"] == {
        "max_tokens": 40, "timeout": 1800, "_roadmap_budget_enforced": True,
    }
    assert scheduled == [session.id]
    assert run.active_blockers == [{"scope": "claim:unrelated"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter,caps,allowed", [
    ("api", {"max_tokens": "100"}, True),
    ("cli", {"max_hours": "0.5"}, True),
    ("cli", {"max_tokens": "100"}, False),
    ("webhook", {"max_hours": "0.5"}, False),
])
async def test_root_outcome_adapter_capability_is_fail_closed(
    db_session, test_project, test_agent, monkeypatch, adapter, caps, allowed,
):
    from sqlalchemy import func, select
    from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
    from huddleroom.models.session import Session
    from huddleroom.services.session_service import SessionClaimAttention

    _goal, run, task, _action = await _root_outcome(db_session, test_project, test_agent, caps)
    scheduled = []
    monkeypatch.setattr(SessionService, "_schedule_dispatch_after_commit", lambda *args: scheduled.append(args[2]))
    request = SessionCreate(agent_id=test_agent.id, project_id=test_project.id, task_id=task.id,
                            adapter_type_override=adapter)
    if allowed:
        run.active_blockers = [{"scope": f"adapter-capability:{run.id}"}, {"scope": "adapter-capability:other"}]
        session = await SessionService().create(db_session, request)
        assert session.metadata_["_run_config"]["_roadmap_budget_enforced"] is True
        assert session.metadata_["_run_config"].get("timeout") == (1800 if adapter == "cli" else None)
        assert run.active_blockers == [{"scope": "adapter-capability:other"}]
    else:
        for _ in range(2):
            with pytest.raises(SessionClaimAttention) as refused:
                await SessionService().create(db_session, request)
            assert refused.value.blocker["scope"] == f"adapter-capability:{run.id}"
            await SessionService.persist_claim_attention(db_session, refused.value)
        assert len(run.active_blockers) == 1
        assert await db_session.scalar(select(func.count()).select_from(Session).where(Session.task_id == task.id)) == 0
        assert not scheduled
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAuthorityDecision)) == 0


async def _planning_task(db_session, test_project, test_agent, *, action_id, metadata_action_id=None):
    task = Task(
        id=uuid.uuid4(), project_id=test_project.id, title="Plan", assigned_to=test_agent.id,
        metadata_={"orchestration": {"action_id": str(metadata_action_id or action_id)}},
    )
    db_session.add(task)
    await db_session.flush()
    return task


@pytest.mark.asyncio
async def test_capped_standalone_plan_claim_stores_remaining_limits_before_dispatch(
    db_session, test_project, test_agent, monkeypatch,
):
    """Removing the metadata-bound plan action would let this capped claim dispatch unbounded."""
    goal = OrchestrationGoal(
        id=uuid.uuid4(), project_id=test_project.id, objective="Plan",
        budget={"caps": {"max_tokens": "100", "max_hours": "1"}},
    )
    run = OrchestrationRun(id=uuid.uuid4(), goal_id=goal.id, status="running", phase="authorized")
    db_session.add(goal)
    await db_session.flush()
    db_session.add(run)
    await db_session.flush()
    action = OrchestrationAction(
        run_id=run.id, idempotency_key="plan", action_type="request_plan", status="completed",
    )
    db_session.add(action)
    await db_session.flush()
    task = await _planning_task(db_session, test_project, test_agent, action_id=action.id)
    action.target_type, action.target_id = "task", task.id
    scheduled = []
    monkeypatch.setattr(SessionService, "_schedule_dispatch_after_commit", lambda *args: scheduled.append(args[2]))

    session = await SessionService().create(
        db_session, SessionCreate(agent_id=test_agent.id, project_id=test_project.id, task_id=task.id),
    )

    assert session.metadata_["_run_config"] == {
        "max_tokens": 100, "timeout": 3600, "_roadmap_budget_enforced": True,
    }
    assert scheduled == [session.id]


@pytest.mark.asyncio
async def test_funded_plan_claim_clamps_session_limits_to_its_allocation(
    db_session, test_project, test_agent, monkeypatch,
):
    goal = OrchestrationGoal(
        id=uuid.uuid4(), project_id=test_project.id, objective="Plan", budget={"caps": {"max_tokens": "9"}},
    )
    run = OrchestrationRun(id=uuid.uuid4(), goal_id=goal.id, status="running", phase="authorized")
    db_session.add_all([goal, run])
    await db_session.flush()
    action = OrchestrationAction(
        run_id=run.id, idempotency_key="plan-funded", action_type="request_plan", status="completed",
        budget_ledger={
            "allocation": {"max_tokens": "4"}, "reserved": {}, "committed": {"max_tokens": "4"},
            "consumed": {}, "usage_state": "known", "enforceability": "enforceable",
        },
    )
    db_session.add(action)
    await db_session.flush()
    task = await _planning_task(db_session, test_project, test_agent, action_id=action.id)
    action.target_type, action.target_id = "task", task.id
    monkeypatch.setattr(SessionService, "_schedule_dispatch_after_commit", lambda *_args: None)

    session = await SessionService().create(
        db_session, SessionCreate(agent_id=test_agent.id, project_id=test_project.id, task_id=task.id),
    )

    assert session.metadata_["_run_config"]["max_tokens"] == 4


@pytest.mark.asyncio
async def test_standalone_plan_claim_rejects_metadata_action_with_wrong_task_lineage(
    db_session, test_project, test_agent,
):
    """Changing the action target must reject the claim instead of selecting another action."""
    goal = OrchestrationGoal(
        id=uuid.uuid4(), project_id=test_project.id, objective="Plan", budget={"caps": {"max_tokens": "100"}},
    )
    run = OrchestrationRun(id=uuid.uuid4(), goal_id=goal.id, status="running", phase="authorized")
    db_session.add(goal)
    await db_session.flush()
    db_session.add(run)
    await db_session.flush()
    action = OrchestrationAction(
        run_id=run.id, idempotency_key="plan", action_type="request_plan", status="completed",
        target_type="task", target_id=uuid.uuid4(),
    )
    db_session.add(action)
    await db_session.flush()
    task = await _planning_task(db_session, test_project, test_agent, action_id=action.id)
    db_session.add(OrchestrationAction(
        run_id=run.id, idempotency_key="other-plan", action_type="request_plan", status="completed",
        target_type="task", target_id=task.id,
    ))
    await db_session.flush()

    with pytest.raises(HTTPException, match="Planning action task lineage is invalid") as exc_info:
        await SessionService().create(
            db_session, SessionCreate(agent_id=test_agent.id, project_id=test_project.id, task_id=task.id),
        )

    assert exc_info.value.status_code == 409
