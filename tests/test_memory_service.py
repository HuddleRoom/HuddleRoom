import uuid
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch
from huddleroom.models.memory_item import MemoryItem
from huddleroom.schemas.memory import MemoryCreate, MemoryResponse, MemorySearchResult, MemorySearchRequest


from huddleroom.services.memory_service import MemoryService

# Re-use fixtures from conftest.py: db_session, test_agent, test_project


@pytest_asyncio.fixture
async def memory_agent(db_session):
    """Agent with memory enabled."""
    from huddleroom.models.agent import Agent
    agent = Agent(
        name=f"memory-agent-{uuid.uuid4()}",
        role="developer",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=[],
        config={"memory_enabled": True},
    )
    db_session.add(agent)
    await db_session.flush()
    return agent


@pytest.fixture
def svc():
    return MemoryService()


@pytest.mark.asyncio
async def test_write_project_memory(db_session, memory_agent, test_project, svc):
    with patch("huddleroom.services.memory_service.embedding_service.generate_embedding", new_callable=AsyncMock, return_value=[0.1] * 10):
        item = await svc.write(
            db=db_session,
            agent_id=memory_agent.id,
            project_id=test_project.id,
            content="We decided to use FastAPI",
            tags=["decision"],
            shared=True,
            scope="project",
        )
    assert item.content == "We decided to use FastAPI"
    assert item.agent_id == memory_agent.id
    assert item.project_id == test_project.id
    assert item.scope == "project"
    assert item.shared is True
    assert item.tags == ["decision"]


@pytest.mark.asyncio
async def test_write_global_memory_sets_null_project(db_session, memory_agent, svc):
    with patch("huddleroom.services.memory_service.embedding_service.generate_embedding", new_callable=AsyncMock, return_value=[0.1] * 10):
        item = await svc.write(
            db=db_session,
            agent_id=memory_agent.id,
            project_id=uuid.uuid4(),  # should be overridden to None
            content="Global lesson",
            tags=[],
            shared=True,
            scope="global",
        )
    assert item.project_id is None
    assert item.scope == "global"


@pytest.mark.asyncio
async def test_read_own_memory(db_session, memory_agent, test_project, svc):
    with patch("huddleroom.services.memory_service.embedding_service.generate_embedding", new_callable=AsyncMock, return_value=[0.1] * 10):
        item = await svc.write(db_session, memory_agent.id, test_project.id, "test", [], True, "project")
    result = await svc.read(db_session, memory_agent.id, test_project.id, item.id)
    assert result is not None
    assert result.id == item.id


@pytest.mark.asyncio
async def test_read_inaccessible_returns_none(db_session, memory_agent, test_project, svc):
    other_agent_id = uuid.uuid4()  # nonexistent agent
    with patch("huddleroom.services.memory_service.embedding_service.generate_embedding", new_callable=AsyncMock, return_value=[0.1] * 10):
        item = await svc.write(db_session, memory_agent.id, test_project.id, "private", [], False, "project")
    # Different agent trying to read private memory
    result = await svc.read(db_session, other_agent_id, test_project.id, item.id)
    assert result is None


@pytest.mark.asyncio
async def test_delete_own_memory(db_session, memory_agent, test_project, svc):
    with patch("huddleroom.services.memory_service.embedding_service.generate_embedding", new_callable=AsyncMock, return_value=[0.1] * 10):
        item = await svc.write(db_session, memory_agent.id, test_project.id, "to delete", [], True, "project")
    deleted = await svc.delete(db_session, memory_agent.id, item.id)
    assert deleted is True
    result = await svc.read(db_session, memory_agent.id, test_project.id, item.id)
    assert result is None


@pytest.mark.asyncio
async def test_delete_other_agents_memory_fails(db_session, memory_agent, test_project, svc):
    with patch("huddleroom.services.memory_service.embedding_service.generate_embedding", new_callable=AsyncMock, return_value=[0.1] * 10):
        item = await svc.write(db_session, memory_agent.id, test_project.id, "not yours", [], True, "project")
    deleted = await svc.delete(db_session, uuid.uuid4(), item.id)
    assert deleted is False



@pytest.mark.asyncio
async def test_search_returns_relevant_memories(db_session, memory_agent, test_project, svc):
    embedding_a = [1.0, 0.0, 0.0, 0.0, 0.0]
    embedding_b = [0.0, 1.0, 0.0, 0.0, 0.0]
    query_embedding = [0.9, 0.1, 0.0, 0.0, 0.0]  # similar to a

    with patch("huddleroom.services.memory_service.embedding_service.generate_embedding", new_callable=AsyncMock, return_value=embedding_a):
        await svc.write(db_session, memory_agent.id, test_project.id, "memory A", ["decision"], True, "project")
    with patch("huddleroom.services.memory_service.embedding_service.generate_embedding", new_callable=AsyncMock, return_value=embedding_b):
        await svc.write(db_session, memory_agent.id, test_project.id, "memory B", ["lesson"], True, "project")

    with patch("huddleroom.services.memory_service.embedding_service.generate_embedding", new_callable=AsyncMock, return_value=query_embedding):
        results = await svc.search(db_session, memory_agent.id, test_project.id, "query", limit=5)

    assert len(results) >= 1
    assert results[0].content == "memory A"
    assert results[0].relevance_score > 0.7


@pytest.mark.asyncio
async def test_search_excludes_inaccessible(db_session, memory_agent, test_project, svc):
    embedding = [1.0, 0.0, 0.0]

    with patch("huddleroom.services.memory_service.embedding_service.generate_embedding", new_callable=AsyncMock, return_value=embedding):
        await svc.write(db_session, memory_agent.id, test_project.id, "private mem", [], False, "project")

    other_agent_id = uuid.uuid4()
    with patch("huddleroom.services.memory_service.embedding_service.generate_embedding", new_callable=AsyncMock, return_value=embedding):
        results = await svc.search(db_session, other_agent_id, test_project.id, "query", limit=5)

    assert len(results) == 0


@pytest.mark.asyncio
async def test_search_filters_by_tags(db_session, memory_agent, test_project, svc):
    embedding = [1.0, 0.0, 0.0]

    with patch("huddleroom.services.memory_service.embedding_service.generate_embedding", new_callable=AsyncMock, return_value=embedding):
        await svc.write(db_session, memory_agent.id, test_project.id, "decision mem", ["decision"], True, "project")
        await svc.write(db_session, memory_agent.id, test_project.id, "lesson mem", ["lesson"], True, "project")

    with patch("huddleroom.services.memory_service.embedding_service.generate_embedding", new_callable=AsyncMock, return_value=embedding):
        results = await svc.search(db_session, memory_agent.id, test_project.id, "query", limit=5, tags=["decision"])

    assert len(results) == 1
    assert results[0].tags == ["decision"]


@pytest.mark.asyncio
async def test_search_embedding_failure_returns_empty(db_session, memory_agent, test_project, svc):
    with patch("huddleroom.services.memory_service.embedding_service.generate_embedding", new_callable=AsyncMock, return_value=None):
        results = await svc.search(db_session, memory_agent.id, test_project.id, "query", limit=5)
    assert results == []


@pytest.mark.asyncio
async def test_project_memory_invisible_from_other_project(db_session, memory_agent, test_project, svc):
    """Agent's project-scoped memory in Project A is invisible from Project B."""
    other_project_id = uuid.uuid4()
    with patch("huddleroom.services.memory_service.embedding_service.generate_embedding", new_callable=AsyncMock, return_value=[1.0, 0.0]):
        item = await svc.write(db_session, memory_agent.id, test_project.id, "project A only", [], True, "project")

    # Same agent, different project — should not see it
    result = await svc.read(db_session, memory_agent.id, other_project_id, item.id)
    assert result is None


@pytest.mark.asyncio
async def test_global_shared_visible_from_any_project(db_session, memory_agent, svc):
    """Global shared memory visible from any project context."""
    with patch("huddleroom.services.memory_service.embedding_service.generate_embedding", new_callable=AsyncMock, return_value=[1.0, 0.0]):
        item = await svc.write(db_session, memory_agent.id, None, "global shared", [], True, "global")

    any_project_id = uuid.uuid4()
    any_agent_id = uuid.uuid4()
    result = await svc.read(db_session, any_agent_id, any_project_id, item.id)
    assert result is not None


@pytest.mark.asyncio
async def test_global_private_only_visible_to_creator(db_session, memory_agent, svc):
    """Global private memory only visible to creating agent."""
    with patch("huddleroom.services.memory_service.embedding_service.generate_embedding", new_callable=AsyncMock, return_value=[1.0, 0.0]):
        item = await svc.write(db_session, memory_agent.id, None, "global private", [], False, "global")

    # Creator sees it
    result = await svc.read(db_session, memory_agent.id, uuid.uuid4(), item.id)
    assert result is not None

    # Other agent does not
    result = await svc.read(db_session, uuid.uuid4(), uuid.uuid4(), item.id)
    assert result is None


@pytest.mark.asyncio
async def test_rest_human_sees_all_project_memories(db_session, memory_agent, test_project, svc):
    """REST (agent_id=None) sees private memories too."""
    with patch("huddleroom.services.memory_service.embedding_service.generate_embedding", new_callable=AsyncMock, return_value=[1.0, 0.0]):
        item = await svc.write(db_session, memory_agent.id, test_project.id, "private", [], False, "project")

    # Human access (agent_id=None) bypasses access rules
    result = await svc.read(db_session, None, test_project.id, item.id)
    assert result is not None


@pytest.mark.asyncio
async def test_embedding_failure_still_saves_memory(db_session, memory_agent, test_project, svc):
    """Memory saved even when embedding generation fails."""
    with patch("huddleroom.services.memory_service.embedding_service.generate_embedding", new_callable=AsyncMock, return_value=None):
        item = await svc.write(db_session, memory_agent.id, test_project.id, "no embedding", [], True, "project")

    assert item.embedding is None
    assert item.content == "no embedding"

    # But it won't appear in search (no embedding to compare)
    with patch("huddleroom.services.memory_service.embedding_service.generate_embedding", new_callable=AsyncMock, return_value=[1.0, 0.0]):
        results = await svc.search(db_session, memory_agent.id, test_project.id, "no embedding", limit=5)
    # Item has no embedding, so cosine similarity can't be computed
    assert all(r.id != item.id for r in results)
