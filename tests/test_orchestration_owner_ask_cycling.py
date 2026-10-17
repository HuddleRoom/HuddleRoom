"""SPR 215: an owner ask that was cancelled or expired is asked again in a new cycle; an answered one stays terminal."""
import json
from contextlib import contextmanager
import uuid
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy import select

from huddleroom.models.orchestration import OrchestrationEvidence
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
from tests.test_orchestration_escalation import _answer, _asks, _make_run
from tests.test_orchestration_review_context import (  # noqa: F401  (fixtures)
    PRODUCER_REPORT, _finish, _owner_asks, _rejection, _revision_tasks, _reverifier, _verifier_for_db_only_report,
    no_live_decisions, workspace,
)

pytestmark = pytest.mark.asyncio


def _blockers(run):
    return [b for b in run.active_blockers if b["kind"] == "owner_escalation"]


@pytest.fixture(autouse=True)
def _no_live_native_dispatch(monkeypatch):
    """Keep committed semantic fixtures from launching real app workers."""
    monkeypatch.setattr("huddleroom.workers.task_runner.register_session", lambda *args, **kwargs: None)


@pytest.fixture
async def test_engine(tmp_path):
    """Isolate committed native setup, including cyclic roadmap references."""
    from sqlalchemy import event
    from huddleroom.models.base import Base
    from tests.conftest import _seed_anon_actors

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'owned-orchestration.db'}")

    @event.listens_for(engine.sync_engine, "connect")
    def foreign_keys(connection, _record):
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        await _seed_anon_actors(connection)
    yield engine
    await engine.dispose()


@pytest.fixture(autouse=True)
async def clean_test_database(test_engine):
    # Per-test database is discarded whole; no global FK cleanup is changed.
    yield


@pytest.fixture
async def db_session(test_engine):
    """This module owns committed setup for real service/worker entrypoints."""
    async with AsyncSession(test_engine, expire_on_commit=False) as db:
        yield db
        await db.rollback()


@contextmanager
def _owned_review_preparation(db):
    """Only imported review setup helpers publish their exact test-owned preflight."""
    from huddleroom.services.orchestration_service import OrchestrationService
    original = OrchestrationService.tick

    async def tick(service, candidate, *args, **kwargs):
        root = candidate.sync_session.get_transaction()
        if (candidate is db and not candidate.in_nested_transaction()
                and (root is None or root.origin.name == "AUTOBEGIN")):
            await candidate.flush()
            await candidate.commit()
        return await original(service, candidate, *args, **kwargs)

    with patch.object(OrchestrationService, "tick", tick):
        yield


async def _cancel(db, action):
    decision = await db.get(OrchestrationAuthorityDecision, action.target_id)
    await OrchestrationAuthorityDecisionService().cancel_decision(db, decision, reason="superseded")


async def test_cancelled_ask_opens_a_new_cycle_and_answered_stays_terminal(db_session, test_project, test_user):
    service, goal, run = await _make_run(db_session, test_project.id)
    args = (db_session, goal, run, "revision_exhausted", "gate:a", ["x"], "Q")
    first, created = await service._escalate_owner_created(*args)
    again, created_again = await service._escalate_owner_created(*args)
    assert created and not created_again and again.id == first.id

    await _cancel(db_session, first)
    second, created = await service._escalate_owner_created(*args)
    assert created and second.id != first.id and second.idempotency_key == first.idempotency_key + ":c1"
    [blocker] = _blockers(run)
    assert blocker["decision_id"] == str(second.target_id)
    assert (await service._latest_owner_ask(db_session, run, "revision_exhausted", "gate:a"))[1] == 1

    await _answer(db_session, second, test_user)
    await db_session.refresh(run)
    third, created = await service._escalate_owner_created(*args)
    assert not created and third.id == second.id and _blockers(run) == []
    assert len(await _asks(db_session, run)) == 2


@pytest.mark.usefixtures("workspace", "no_live_decisions", "safe_effectiveness_review_continue")
async def test_cancelled_revision_exhausted_ask_is_asked_again(db_session, test_project):
    from huddleroom.models.orchestration import OrchestrationGoal

    db = db_session
    with _owned_review_preparation(db):
        service, run, gate, work_task, _artifact, _verifier, verifier_task = await _verifier_for_db_only_report(db, test_project)
        await _finish(db, service, run, verifier_task, _rejection("Empty password returns 500."))
        [revision] = await _revision_tasks(db, work_task)
        await _finish(db, service, run, revision, json.dumps(PRODUCER_REPORT))
        await _finish(db, service, run, await _reverifier(db, service, run, revision), _rejection("Empty password returns 500."))
    [ask] = _owner_asks(run, "revision_exhausted")
    first = (await _asks(db, run))[-1]
    assert str(first.target_id) == ask["decision_id"]
    await _cancel(db, first)

    goal = await db.get(OrchestrationGoal, run.goal_id)
    rejected = (await db.scalars(select(OrchestrationEvidence).where(
        OrchestrationEvidence.gate_id == gate.id, OrchestrationEvidence.verdict == "rejected",
    ).order_by(OrchestrationEvidence.created_at.desc()))).first()
    assert await service._request_gate_revision(db, goal, run, gate, "verification_rejected", rejected) == 1
    [ask] = _owner_asks(run, "revision_exhausted")
    assert ask["decision_id"] != str(first.target_id)
    assert any(a.idempotency_key.endswith(":c1") for a in await _asks(db, run))
    assert await service._request_gate_revision(db, goal, run, gate, "verification_rejected", rejected) == 0
    assert len(_owner_asks(run, "revision_exhausted")) == 1


async def test_budget_ask_cancel_race_still_drops_the_blocker(db_session, test_project):
    service, goal, run = await _make_run(db_session, test_project.id)
    await service._escalate_owner_created(db_session, goal, run, "budget_exhausted", "gate:b", ["x"], "Q")
    assert _blockers(run)

    async def boom(*_a, **_k):
        raise ValueError("already resolved")

    with patch.object(OrchestrationAuthorityDecisionService, "cancel_decision", boom):
        await service._resolve_budget_ask(db_session, run, "gate:b", "funded")
    assert _blockers(run) == []


@pytest.mark.usefixtures("workspace", "no_live_decisions", "safe_effectiveness_review_continue")
async def test_cancelled_reverification_capability_ask_re_arms(db_session, test_project):
    from pathlib import Path

    from tests.test_orchestration_result_correction import _complete
    from tests.test_orchestration_review_context import ACCEPTED, _verification_actions

    db = db_session
    with _owned_review_preparation(db):
        service, run, gate, _work_task, _artifact, _verifier, verifier_task = await _verifier_for_db_only_report(db, test_project)
    (Path(test_project.workspace_path) / "login.py").write_text("def login(email, password): return True\n")
    await _complete(db, service, run, verifier_task, ACCEPTED)
    with patch.object(service, "_best_recovery_agent", AsyncMock(return_value=None)):
        for _ in range(4):
            await db.flush()
            await db.commit()
            await service.tick(db, run.id)
    [ask] = _owner_asks(run, "capability_unavailable")
    first = next(a for a in await _asks(db, run) if str(a.target_id) == ask["decision_id"])
    await _cancel(db, first)

    await db.flush()
    await db.commit()
    await service.tick(db, run.id)  # re-armed: a fresh attempt is dispatched, earlier failures no longer count
    actions = await _verification_actions(db, run, gate)
    assert len(actions) == 5 and actions[-1].status == "completed" and actions[-1].target_type == "task"
