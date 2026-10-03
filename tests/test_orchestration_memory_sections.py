import uuid

import pytest
import pytest_asyncio
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError


@pytest_asyncio.fixture
async def orch_goal(db_session, test_project):
    from huddleroom.models.orchestration import OrchestrationGoal

    goal = OrchestrationGoal(
        project_id=test_project.id,
        objective="Ship the widget",
        success_criteria=[{"key": "works", "description": "widget works"}],
    )
    db_session.add(goal)
    await db_session.flush()
    return goal


@pytest.mark.asyncio
async def test_memory_section_insert_defaults(db_session, test_project, orch_goal):
    from huddleroom.models.orchestration_memory import OrchestrationMemorySection

    section = OrchestrationMemorySection(
        project_id=test_project.id,
        goal_id=orch_goal.id,
        section_key="goal_definition",
        title="Goal definition",
        body="The goal is to ship the widget.",
        created_by="orchestrator",
    )
    db_session.add(section)
    await db_session.flush()

    assert section.id is not None
    assert section.section_type == "text"
    assert section.always_load is False or section.always_load == 0  # SQLite may return int
    assert section.toc_order == 0
    assert section.run_id is None
    assert section.summary is None
    assert section.created_at is not None
    assert section.updated_at is not None


@pytest.mark.asyncio
async def test_memory_section_key_unique_per_goal(db_session, test_project, orch_goal):
    from huddleroom.models.orchestration_memory import OrchestrationMemorySection

    def make():
        return OrchestrationMemorySection(
            project_id=test_project.id,
            goal_id=orch_goal.id,
            section_key="constraints_tradeoffs",
            title="Constraints",
            body="No cloud dependencies.",
            created_by="orchestrator",
        )

    db_session.add(make())
    await db_session.flush()
    db_session.add(make())
    with pytest.raises(IntegrityError):
        await db_session.flush()


@pytest.mark.asyncio
async def test_service_upsert_creates_then_updates(db_session, test_project, orch_goal):
    from huddleroom.services.orchestration_memory_service import OrchestrationMemoryService

    svc = OrchestrationMemoryService()
    created = await svc.upsert_section(
        db_session,
        test_project.id,
        orch_goal.id,
        section_key="goal_definition",
        title="Goal definition",
        body="v1",
        created_by="orchestrator:goal_definition",
    )
    assert created.body == "v1"
    assert created.created_by == "orchestrator:goal_definition"

    updated = await svc.upsert_section(
        db_session,
        test_project.id,
        orch_goal.id,
        section_key="goal_definition",
        title="Goal definition",
        body="v2",
        created_by="human:1234",
    )
    assert updated.id == created.id
    assert updated.body == "v2"
    assert updated.created_by == "human:1234"


@pytest.mark.asyncio
async def test_service_rejects_bad_section_key(db_session, test_project, orch_goal):
    from huddleroom.services.orchestration_memory_service import OrchestrationMemoryService

    svc = OrchestrationMemoryService()
    with pytest.raises(ValueError):
        await svc.upsert_section(
            db_session, test_project.id, orch_goal.id,
            section_key="Bad Key!", title="x", body="y",
        )


@pytest.mark.asyncio
async def test_service_get_section_scoped_to_project_and_goal(db_session, test_project, orch_goal):
    from huddleroom.models.project import Project
    from huddleroom.services.orchestration_memory_service import OrchestrationMemoryService

    other = Project(name="Other Project", description="", config={})
    db_session.add(other)
    await db_session.flush()

    svc = OrchestrationMemoryService()
    await svc.upsert_section(
        db_session, test_project.id, orch_goal.id,
        section_key="introduction", title="Intro", body="hello",
    )
    assert await svc.get_section(db_session, test_project.id, orch_goal.id, "introduction") is not None
    assert await svc.get_section(db_session, other.id, orch_goal.id, "introduction") is None
    assert await svc.get_section(db_session, test_project.id, uuid.uuid4(), "introduction") is None


@pytest.mark.asyncio
async def test_service_always_loaded_and_toc_ordering(db_session, test_project, orch_goal):
    from huddleroom.services.orchestration_memory_service import OrchestrationMemoryService

    svc = OrchestrationMemoryService()
    await svc.upsert_section(
        db_session, test_project.id, orch_goal.id,
        section_key="lessons_learned", title="Lessons", body="none yet", toc_order=5,
    )
    await svc.upsert_section(
        db_session, test_project.id, orch_goal.id,
        section_key="introduction", title="Intro", body="hello",
        toc_order=0, always_load=True,
    )

    sections = await svc.list_sections(db_session, test_project.id, orch_goal.id)
    assert [s.section_key for s in sections] == ["introduction", "lessons_learned"]

    preface = await svc.get_always_loaded(db_session, test_project.id, orch_goal.id)
    assert [s.section_key for s in preface] == ["introduction"]


@pytest.mark.asyncio
async def test_memory_api_crud(client, test_project, orch_goal):
    base = f"/api/v1/projects/{test_project.id}/orchestration/goals/{orch_goal.id}/memory"

    resp = await client.post(base, json={
        "section_key": "goal_definition",
        "title": "Goal definition",
        "body": "Ship the widget by Friday.",
        "summary": "Ship widget",
        "always_load": True,
        "toc_order": 1,
    })
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["section_key"] == "goal_definition"
    assert payload["created_by"].startswith("human:")

    resp = await client.get(base)
    assert resp.status_code == 200
    overview = resp.json()
    assert [entry["section_key"] for entry in overview["toc"]] == ["goal_definition"]
    assert "body" not in overview["toc"][0]  # TOC is compact
    assert overview["always_loaded"][0]["body"] == "Ship the widget by Friday."

    resp = await client.get(f"{base}/goal_definition")
    assert resp.status_code == 200
    assert resp.json()["body"] == "Ship the widget by Friday."

    resp = await client.get(f"{base}/missing_section")
    assert resp.status_code == 404

    resp = await client.post(base, json={
        "section_key": "goal_definition",
        "title": "Goal definition",
        "body": "Ship the widget by Thursday.",
    })
    assert resp.status_code == 200
    assert resp.json()["id"] == payload["id"]  # upsert updated in place


@pytest.mark.asyncio
async def test_memory_api_rejects_trailing_newline_and_overlength_key(client, test_project, orch_goal):
    base = f"/api/v1/projects/{test_project.id}/orchestration/goals/{orch_goal.id}/memory"
    resp = await client.post(base, json={
        "section_key": "goal_definition\n",
        "title": "x",
        "body": "y",
    })
    assert resp.status_code == 422

    resp = await client.post(base, json={
        "section_key": "a" * 101,
        "title": "x",
        "body": "y",
    })
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_memory_api_rejects_oversized_toc_order(client, test_project, orch_goal):
    base = f"/api/v1/projects/{test_project.id}/orchestration/goals/{orch_goal.id}/memory"
    resp = await client.post(base, json={
        "section_key": "goal_definition",
        "title": "x",
        "body": "y",
        "toc_order": 2_147_483_648,
    })
    assert resp.status_code == 422

    resp = await client.post(base, json={
        "section_key": "goal_definition",
        "title": "x",
        "body": "y",
        "toc_order": -1,
    })
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_memory_api_rejects_client_run_id(client, test_project, orch_goal):
    base = f"/api/v1/projects/{test_project.id}/orchestration/goals/{orch_goal.id}/memory"
    resp = await client.post(base, json={
        "section_key": "goal_definition",
        "title": "Goal definition",
        "body": "Ship the widget by Friday.",
        "run_id": str(uuid.uuid4()),
    })
    assert resp.status_code == 422

    resp = await client.get(base)
    assert resp.status_code == 200
    assert resp.json()["toc"] == []


@pytest.mark.asyncio
async def test_memory_api_unknown_goal_404(client, test_project):
    base = f"/api/v1/projects/{test_project.id}/orchestration/goals/{uuid.uuid4()}/memory"
    resp = await client.get(base)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_memory_api_rejects_bad_section_key(client, test_project, orch_goal):
    base = f"/api/v1/projects/{test_project.id}/orchestration/goals/{orch_goal.id}/memory"
    resp = await client.post(base, json={
        "section_key": "Bad Key!",
        "title": "x",
        "body": "y",
    })
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_no_agent_facing_orchestrator_memory_route():
    from huddleroom.main import create_app

    paths = create_app().openapi()["paths"]
    memory_paths = [path for path in paths if "memory" in path]
    orch_memory_paths = [p for p in memory_paths if "orchestration" in p]
    assert orch_memory_paths, "orchestrator memory routes must exist"
    for path in orch_memory_paths:
        assert path.startswith("/api/v1/projects/"), f"orchestrator memory leaked outside project scope: {path}"
    agent_paths = [p for p in memory_paths if p.startswith("/api/v1/agent")]
    assert all("orchestration" not in p for p in agent_paths)


@pytest.mark.asyncio
async def test_orchestrator_memory_separate_from_agent_memory(db_session, test_project, orch_goal):
    from sqlalchemy import func, select

    from huddleroom.models.memory_item import MemoryItem
    from huddleroom.services.orchestration_memory_service import OrchestrationMemoryService

    before = (await db_session.execute(select(func.count()).select_from(MemoryItem))).scalar()
    await OrchestrationMemoryService().upsert_section(
        db_session, test_project.id, orch_goal.id,
        section_key="introduction", title="Intro", body="orchestrator-only",
    )
    after = (await db_session.execute(select(func.count()).select_from(MemoryItem))).scalar()
    assert after == before  # writes never touch agent memory
