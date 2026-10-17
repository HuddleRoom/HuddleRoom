from decimal import Decimal

from sqlalchemy import select

import pytest
import pytest_asyncio

from huddleroom.models.orchestration import OrchestrationAction
from huddleroom.services.orchestration_budget_service import OrchestrationBudgetService
from tests.test_orchestration_roadmap_task_items import accepted_roadmap as accepted_roadmap_fixture
from tests.test_orchestration_roadmap_task_items import roadmap_task
from huddleroom.services.orchestration_service import OrchestrationService

@pytest_asyncio.fixture
async def test_engine(tmp_path):
    """SPR230: isolate committed preparation used by real native ticks."""
    from sqlalchemy import event
    from sqlalchemy.ext.asyncio import create_async_engine
    from huddleroom.models.base import Base
    from tests.conftest import _seed_anon_actors

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'summary-owned-setup.db'}")
    @event.listens_for(engine.sync_engine, "connect")
    def pragmas(connection, _record):
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        await _seed_anon_actors(connection)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture(autouse=True)
async def clean_test_database(test_engine):
    # The per-test database is discarded; cyclic committed rows need no DELETE cleanup.
    yield


@pytest_asyncio.fixture
async def db_session(test_engine, monkeypatch):
    from sqlalchemy.ext.asyncio import AsyncSession

    # SPR230: committed fixtures must never launch application agents.
    monkeypatch.setattr("huddleroom.workers.task_runner.register_session", lambda *_args, **_kwargs: None)

    async with AsyncSession(test_engine, expire_on_commit=False) as db:
        yield db
        await db.rollback()





@pytest.fixture(autouse=True)
def _manual_task_start(monkeypatch):
    # These tests drive the release-then-manual-run flow; auto-start is covered in test_orchestration_task_autostart.py.
    async def _noop(self, db, goal, run):
        return 0
    monkeypatch.setattr(OrchestrationService, "_start_released_tasks", _noop)


@pytest.mark.asyncio
async def test_independent_roadmap_tasks_share_discretionary_budget(db_session, test_project, monkeypatch):
    async def setup(_planner, goal, _run):
        goal.budget = {"caps": {"max_tokens": 9, "max_hours": 1}}

    accepted_roadmap = await accepted_roadmap_fixture.__wrapped__(db_session, test_project)
    original_tick = OrchestrationService.tick

    async def setup_tick(service, db, *args, **kwargs):
        # SPR230: only this imported fixture's authored setup owns the transaction.
        root = db.sync_session.get_transaction()
        if db is db_session and not db.in_nested_transaction() and (root is None or root.origin.name == "AUTOBEGIN"):
            await db.commit()
        return await original_tick(service, db, *args, **kwargs)

    with monkeypatch.context() as setup_patch:
        setup_patch.setattr(OrchestrationService, "tick", setup_tick)
        service, _goal, run, _verifier = await accepted_roadmap(
            [roadmap_task("first"), roadmap_task("second")], setup=setup,
        )

    await db_session.commit()  # SPR230: publish this test-owned preparation.
    await service.tick(db_session, run.id)
    await db_session.commit()  # SPR230: publish this test-owned preparation.
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
