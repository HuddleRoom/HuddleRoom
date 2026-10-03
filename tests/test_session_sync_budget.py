import uuid

import pytest

from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGoal, OrchestrationRun
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.services.orchestration_budget_service import OrchestrationBudgetService
from huddleroom.services.session_sync import sync_task_from_session


pytestmark = pytest.mark.asyncio


async def test_provisional_session_settlement_keeps_its_scoped_measurement_blocker(
    db_session, test_project, test_agent,
):
    goal = OrchestrationGoal(
        project_id=test_project.id, objective="budget", original_request="budget",
        budget={"caps": {"max_tokens": 100}},
    )
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id, status="running", phase="authorized")
    task = Task(project_id=test_project.id, title="budgeted task", status="in_progress")
    db_session.add_all([run, task])
    await db_session.flush()
    action = OrchestrationAction(
        run_id=run.id, idempotency_key=f"sync-budget:{uuid.uuid4()}",
        action_type="create_delegation_task", request={}, target_type="task", target_id=task.id,
        status="completed", budget_ledger={
            "allocation": {"max_tokens": "10"}, "reserved": {}, "committed": {"max_tokens": "10"},
            "consumed": {}, "usage_state": "known", "enforceability": "enforceable",
        },
    )
    db_session.add(action)
    await db_session.flush()
    session = Session(
        task_id=task.id, agent_id=test_agent.id, project_id=test_project.id, adapter_type="test",
        status="completed", metadata_={
            "token_count_in": 2, "token_count_out": 3, "token_usage_complete": False,
            "orchestration": {"action_id": str(action.id)},
        },
    )
    unrelated = {"kind": "budget_measurement", "scope": "action:unrelated:session:unrelated"}
    run.active_blockers = [unrelated]
    db_session.add(session)
    await db_session.flush()

    await sync_task_from_session(db_session, session)
    await sync_task_from_session(db_session, session)

    scope = f"action:{action.id}:session:{session.id}"
    assert [blocker["scope"] for blocker in run.active_blockers if blocker["kind"] == "budget_measurement"] == [
        "action:unrelated:session:unrelated", scope,
    ]
    assert not await OrchestrationBudgetService().can_dispatch(db_session, goal, run, {"max_tokens": 1})
    assert all(blocker.get("scope") for blocker in run.active_blockers if blocker["kind"] == "budget_measurement")

    session.metadata_ = {**session.metadata_, "token_usage_complete": True}
    await sync_task_from_session(db_session, session)

    assert run.active_blockers == [unrelated]
