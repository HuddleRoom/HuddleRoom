from decimal import Decimal

from sqlalchemy import select

import pytest

from huddleroom.models.orchestration import OrchestrationAction
from huddleroom.services.orchestration_budget_service import OrchestrationBudgetService
from tests.test_orchestration_roadmap_task_items import accepted_roadmap as accepted_roadmap_fixture
from tests.test_orchestration_roadmap_task_items import roadmap_task


@pytest.mark.asyncio
async def test_independent_roadmap_tasks_share_discretionary_budget(db_session, test_project):
    async def setup(_planner, goal, _run):
        goal.budget = {"caps": {"max_tokens": 9, "max_hours": 1}}

    accepted_roadmap = await accepted_roadmap_fixture.__wrapped__(db_session, test_project)
    service, _goal, run, _verifier = await accepted_roadmap(
        [roadmap_task("first"), roadmap_task("second")], setup=setup,
    )

    await service.tick(db_session, run.id)
    await service.tick(db_session, run.id)

    actions = (await db_session.scalars(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id,
        OrchestrationAction.action_type == "create_delegation_task",
        OrchestrationAction.idempotency_key.like("%roadmap_item%"),
    ).order_by(OrchestrationAction.created_at))).all()

    assert [action.budget_ledger["allocation"] for action in actions] == [
        {"max_tokens": "4", "max_hours": "0.4998611111111111111111111111"},
        {"max_tokens": "4", "max_hours": "0.4998611111111111111111111111"},
    ]


def test_hour_budget_keeps_a_positive_discretionary_and_protected_quantum():
    service = OrchestrationBudgetService()
    snapshot = service.snapshot(type("Goal", (), {"budget": {"caps": {"max_hours": 1}}})())
    discretionary = type("Action", (), {"budget_ledger": {}})()
    protected = type("Action", (), {"budget_ledger": {}})()

    service.reserve(snapshot, discretionary, {"max_hours": "0.9997222222222222222222222222"})
    service.reserve(snapshot, protected, {"max_hours": "0.0002777777777777777777777778"}, closeout=True)

    assert discretionary.budget_ledger["allocation"]["max_hours"] != "0"
    assert protected.budget_ledger["allocation"]["max_hours"] == "0.0002777777777777777777777778"
    assert sum(Decimal(action.budget_ledger["allocation"]["max_hours"]) for action in (discretionary, protected)) <= 1
