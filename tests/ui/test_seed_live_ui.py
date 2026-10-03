from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from huddleroom.models.base import Base
from huddleroom.models.agent import Agent
from huddleroom.models.event_log import EventLog
from huddleroom.models.hook import Hook
from huddleroom.models.knowledge_item import KnowledgeItem
from huddleroom.models.meeting import (
    Meeting,
    MeetingActionItem,
    MeetingAgendaItem,
    MeetingDecision,
    MeetingEvent,
    MeetingParticipantSignal,
    MeetingTurn,
)
from huddleroom.models.memory_item import MemoryItem
from huddleroom.models.optimization import CostMetric, Optimization, Pattern
from huddleroom.models.project import Project
from huddleroom.models.protocol import Protocol, ProtocolInstance, ProtocolTimeout, ProtocolTransition
from huddleroom.models.routing_rule import RoutingRule
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.models.user import User
import huddleroom.models  # noqa: F401

from tests.ui.seed_live_ui import SEED_AGENT_NAMES, SEED_NAMES, seed_database


SEEDED_COUNTS = {
    Project: 1,
    User: 1,
    Agent: 3,
    Task: 5,
    Session: 2,
    Meeting: 2,
    MeetingAgendaItem: 2,
    MeetingTurn: 3,
    MeetingDecision: 1,
    MeetingActionItem: 1,
    MeetingEvent: 3,
    MeetingParticipantSignal: 2,
    Protocol: 2,
    ProtocolInstance: 2,
    ProtocolTransition: 2,
    ProtocolTimeout: 1,
    KnowledgeItem: 2,
    MemoryItem: 2,
    RoutingRule: 2,
    Hook: 2,
    Pattern: 1,
    Optimization: 3,
    CostMetric: 3,
    EventLog: 3,
}


async def _create_schema(database_url: str) -> None:
    engine = create_async_engine(database_url, echo=False)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    finally:
        await engine.dispose()


async def _count_rows(session, model) -> int:
    return await session.scalar(select(func.count()).select_from(model))


@pytest.mark.asyncio
async def test_seed_database_creates_deterministic_dashboard_records(tmp_path: Path) -> None:
    db_path = tmp_path / "ui-seed.db"
    database_url = f"sqlite+aiosqlite:///{db_path}"
    workspace_dir = tmp_path / "workspace"

    await _create_schema(database_url)

    summary = await seed_database(database_url=database_url, workspace_dir=workspace_dir)

    assert summary["database_url"] == database_url
    assert summary["project_name"] == SEED_NAMES["project"]
    assert summary["workspace_dir"] == str((workspace_dir / "huddleroom-ui-live-project").resolve())

    engine = create_async_engine(database_url, echo=False)
    try:
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as session:
            project = (
                await session.execute(select(Project).where(Project.name == SEED_NAMES["project"]))
            ).scalar_one()
            assert project.workspace_path == str((workspace_dir / "huddleroom-ui-live-project").resolve())

            actual_counts = {
                model.__tablename__: await _count_rows(session, model)
                for model in SEEDED_COUNTS
            }

            assert summary["counts"] == actual_counts
            assert actual_counts == {model.__tablename__: count for model, count in SEEDED_COUNTS.items()}
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_seed_database_reseeding_is_idempotent_and_preserves_non_seed_local_lookalikes(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "ui-seed-idempotent.db"
    database_url = f"sqlite+aiosqlite:///{db_path}"
    workspace_dir = tmp_path / "workspace"

    await _create_schema(database_url)

    first = await seed_database(database_url=database_url, workspace_dir=workspace_dir)

    engine = create_async_engine(database_url, echo=False)
    try:
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as session:
            extra_project = Project(name="Developer Local Project", config={})
            local_lookalike_agent = Agent(
                name="huddleroom-ui-local-dev",
                role="developer",
                provider="openai",
                model="gpt-4o-mini",
                adapter_type="api",
                capabilities=[],
                config={},
                is_active=True,
            )
            session.add_all([extra_project, local_lookalike_agent])
            await session.commit()

        second = await seed_database(database_url=database_url, workspace_dir=workspace_dir)

        async with session_factory() as session:
            seeded_projects = (
                await session.execute(select(Project).where(Project.name == SEED_NAMES["project"]))
            ).scalars().all()
            assert len(seeded_projects) == 1

            local_projects = (
                await session.execute(select(Project).where(Project.name == "Developer Local Project"))
            ).scalars().all()
            assert len(local_projects) == 1

            local_lookalikes = (
                await session.execute(select(Agent).where(Agent.name == "huddleroom-ui-local-dev"))
            ).scalars().all()
            assert len(local_lookalikes) == 1

            actual_counts = {
                model.__tablename__: await _count_rows(session, model)
                for model in SEEDED_COUNTS
            }

            assert second["counts"] == actual_counts
            assert actual_counts["projects"] == SEEDED_COUNTS[Project] + 1
            assert actual_counts["agents"] == SEEDED_COUNTS[Agent] + 1
            for model, expected_count in SEEDED_COUNTS.items():
                if model not in {Project, Agent}:
                    assert actual_counts[model.__tablename__] == expected_count
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_seed_database_rejects_implicit_or_local_default_targets(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="database_url"):
        await seed_database(database_url="", workspace_dir=tmp_path / "workspace")

    with pytest.raises(ValueError, match="sqlite file URL"):
        await seed_database(
            database_url="postgresql+asyncpg://user:pass@localhost/rally",
            workspace_dir=tmp_path / "workspace",
        )

    with pytest.raises(ValueError, match="sqlite file URL"):
        await seed_database(
            database_url="sqlite+aiosqlite:///:memory:",
            workspace_dir=tmp_path / "workspace",
        )

    with pytest.raises(ValueError, match="workspace_dir"):
        await seed_database(
            database_url=f"sqlite+aiosqlite:///{tmp_path / 'ok.db'}",
            workspace_dir=None,
        )

    with pytest.raises(ValueError, match="huddleroom.db"):
        await seed_database(
            database_url="sqlite+aiosqlite:///huddleroom.db",
            workspace_dir=tmp_path / "workspace",
        )

    with pytest.raises(ValueError, match="huddleroom.db"):
        await seed_database(
            database_url="sqlite+aiosqlite:///./huddleroom.db",
            workspace_dir=tmp_path / "workspace",
        )

    for name in ("huddleroom.db", "rally.db"):
        with pytest.raises(ValueError, match=name):
            await seed_database(
                database_url=f"sqlite+aiosqlite:////{(Path.cwd() / name).as_posix().lstrip('/')}",
                workspace_dir=tmp_path / "workspace",
            )


@pytest.mark.asyncio
async def test_seed_database_aborts_on_unmarked_reserved_agent_name_collision(tmp_path: Path) -> None:
    db_path = tmp_path / "ui-seed-collision.db"
    database_url = f"sqlite+aiosqlite:///{db_path}"
    workspace_dir = tmp_path / "workspace"

    await _create_schema(database_url)

    engine = create_async_engine(database_url, echo=False)
    try:
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as session:
            session.add(
                Agent(
                    name=SEED_AGENT_NAMES[0],
                    role="developer",
                    provider="openai",
                    model="gpt-4o-mini",
                    adapter_type="api",
                    capabilities=[],
                    config={},
                    is_active=True,
                )
            )
            await session.commit()

        with pytest.raises(ValueError, match="reserved seed agent name"):
            await seed_database(database_url=database_url, workspace_dir=workspace_dir)

        async with session_factory() as session:
            colliding_agents = (
                await session.execute(select(Agent).where(Agent.name == SEED_AGENT_NAMES[0]))
            ).scalars().all()
            seeded_projects = (
                await session.execute(select(Project).where(Project.name == SEED_NAMES["project"]))
            ).scalars().all()

            assert len(colliding_agents) == 1
            assert colliding_agents[0].config == {}
            assert seeded_projects == []
    finally:
        await engine.dispose()
