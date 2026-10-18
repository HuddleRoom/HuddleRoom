import pytest
from sqlalchemy import select

from huddleroom.models.orchestration_process import OrchestrationWarning
from huddleroom.schemas.orchestration import OrchestrationDecisionAnswerRequest
from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
from huddleroom.services.orchestration_decision_application import answer_decision
from huddleroom.services.orchestration_team_hierarchy import TeamHierarchyProcess
from huddleroom.services.orchestration_team_hierarchy_analyzer import TeamHierarchyAnalyzer
from huddleroom.services.orchestration_warning_service import OrchestrationWarningService
from tests.test_orchestration_team_hierarchy import (
    _analysis_for, _answer_url, _lifecycle_goal, _proposed_agent, _settle_answer,
)

DETAIL = "team hierarchy proposal inputs changed"


@pytest.fixture
async def test_engine(tmp_path):
    from sqlalchemy import event
    from sqlalchemy.ext.asyncio import create_async_engine
    from huddleroom.models.base import Base
    from tests.conftest import _seed_anon_actors
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'stale-answer.db'}")

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


@pytest.fixture(autouse=True)
async def clean_test_database():
    yield


@pytest.fixture
async def db_session(test_engine, monkeypatch):
    from sqlalchemy.ext.asyncio import AsyncSession
    monkeypatch.setattr("huddleroom.workers.task_runner.register_session", lambda *_args, **_kwargs: None)
    async with AsyncSession(test_engine, expire_on_commit=False) as db:
        yield db
        await db.rollback()


async def _stale_decision(db_session, test_project, test_user, monkeypatch):
    goal, run, _agents, _review = await _lifecycle_goal(db_session, test_project, test_user, weight="substantial")

    async def review(_self, payload, project=None, *, project_id=None):
        return _analysis_for(payload, proposals=(_proposed_agent(),))

    monkeypatch.setattr(TeamHierarchyAnalyzer, "review", review)
    await TeamHierarchyProcess().advance(db_session, goal, run)
    decision = (await OrchestrationAuthorityDecisionService().list_decisions(db_session, goal.id))[0]
    goal.objective = "Implement a completely different API"
    await db_session.flush()
    return goal, run, decision


async def _stale_warnings(test_engine, session_cls, goal_id):
    from sqlalchemy.ext.asyncio import async_sessionmaker

    async with async_sessionmaker(test_engine, class_=session_cls, expire_on_commit=False)() as fresh:
        rows = (await fresh.execute(select(OrchestrationWarning).where(
            OrchestrationWarning.goal_id == goal_id,
            OrchestrationWarning.warning_type == "team_hierarchy_stale_inputs"))).scalars().all()
    return rows


@pytest.mark.asyncio
async def test_non_owned_caller_gets_409_and_no_warning(
    db_session, test_project, test_user, monkeypatch, test_engine
):
    from fastapi import HTTPException

    goal, _run, decision = await _stale_decision(db_session, test_project, test_user, monkeypatch)
    goal_id, project_id, decision_id = goal.id, test_project.id, decision.id
    await db_session.commit()
    assert await _stale_warnings(test_engine, type(db_session), goal_id) == []  # precondition
    await db_session.execute(select(type(goal).id))  # read: autobegins a transaction the caller owns
    assert db_session.in_transaction()

    with pytest.raises(HTTPException) as caught:
        await answer_decision(db_session, project_id, goal_id, decision_id,
                              OrchestrationDecisionAnswerRequest(selected_option="approve"), test_user)
    assert caught.value.status_code == 409
    assert caught.value.detail == DETAIL
    await db_session.rollback()

    assert await _stale_warnings(test_engine, type(db_session), goal_id) == []


@pytest.mark.asyncio
async def test_suggestion_failure_keeps_original_409(
    client, db_session, test_project, test_user, monkeypatch, test_engine
):
    goal, run, decision = await _stale_decision(db_session, test_project, test_user, monkeypatch)
    goal_id = goal.id

    async def boom(*_args, **_kwargs):
        raise RuntimeError("warning store down")

    monkeypatch.setattr(OrchestrationWarningService, "suggest_stale_inputs", boom)
    await db_session.commit()
    response = await client.post(_answer_url(test_project.id, goal_id, decision.id),
                                 json={"selected_option": "approve"})
    await _settle_answer(db_session, run, response)

    assert response.status_code == 409
    assert response.json()["detail"] == DETAIL
    assert await _stale_warnings(test_engine, type(db_session), goal_id) == []
