import uuid

import pytest
from sqlalchemy import DateTime, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from huddleroom.models.agent import Agent
from huddleroom.models.base import Base, _utcnow_naive
from huddleroom.models.hook import Hook
from huddleroom.models.meeting import MeetingParticipantSignal, MeetingRequest
from huddleroom.models.orchestration import OrchestrationGoal
from huddleroom.models.project import Project
from huddleroom.models.routing_rule import RoutingRule
from huddleroom.models.user import User


@pytest.fixture(autouse=True)
def clean_test_database():
    """This regression creates its own disposable PostgreSQL schema."""
    yield


def test_naive_utc_default_matches_timestamp_without_timezone() -> None:
    assert _utcnow_naive().tzinfo is None


@pytest.mark.parametrize(
    ("model", "column_names"),
    (
        (RoutingRule, ("created_at", "updated_at")),
        (Hook, ("created_at", "updated_at")),
        (MeetingParticipantSignal, ("created_at",)),
        (MeetingRequest, ("created_at",)),
    ),
)
def test_aware_timestamp_mappings_match_migrations(model, column_names) -> None:
    for column_name in column_names:
        column = model.__table__.c[column_name]
        assert isinstance(column.type, DateTime)
        assert column.type.timezone is True
        assert column.default.arg(None).tzinfo is not None


@pytest.mark.asyncio
@pytest.mark.unsupported_mode
async def test_postgresql_timestamp_defaults_match_column_timezone_contracts(
):
    """asyncpg must receive naive values for timestamp columns and aware values for timestamptz."""
    from huddleroom.config import settings

    if not settings.database_url.startswith("postgresql"):
        pytest.skip("requires PostgreSQL/asyncpg")

    schema = f"timestamp_defaults_{uuid.uuid4().hex}"
    engine = create_async_engine(
        settings.database_url,
        execution_options={"schema_translate_map": {None: schema}},
    )
    tables = [User.__table__, Agent.__table__, Project.__table__, OrchestrationGoal.__table__]
    try:
        async with engine.begin() as conn:
            await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
            await conn.run_sync(lambda sync_conn: Base.metadata.create_all(sync_conn, tables=tables))

        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as db_session:
            agent = Agent(
                name=f"timestamp-default-{uuid.uuid4()}",
                role="developer",
                provider="local",
                model="local",
                adapter_type="api",
                capabilities=[],
                config={},
            )
            project = Project(name="Timestamp defaults")
            db_session.add_all((agent, project))
            await db_session.flush()
            await db_session.refresh(agent)
            assert agent.created_at.tzinfo is None
            assert agent.updated_at.tzinfo is None

            goal = OrchestrationGoal(
                project_id=project.id,
                objective="Keep timezone-aware orchestration timestamps aware",
                success_criteria=[],
                constraints={},
                budget={},
            )
            db_session.add(goal)
            await db_session.flush()
            await db_session.refresh(goal)
            assert goal.created_at.tzinfo is not None
            assert goal.updated_at.tzinfo is not None
    finally:
        async with engine.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await engine.dispose()
