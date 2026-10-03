import uuid

import pytest

from fastapi import HTTPException

from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGoal, OrchestrationRun
from huddleroom.models.task import Task
from huddleroom.schemas.session import SessionCreate
from huddleroom.services.session_service import SessionService


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
