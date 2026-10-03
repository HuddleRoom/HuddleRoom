import uuid
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_api_adapter_injects_memory_context(db_session, test_project):
    """Memory-enabled agent gets memory context injected into messages."""
    from huddleroom.models.agent import Agent
    from huddleroom.models.task import Task
    from huddleroom.adapters.api_adapter import ApiAdapter

    agent = Agent(
        name=f"mem-agent-{uuid.uuid4()}",
        role="developer",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=[],
        config={"memory_enabled": True},
    )
    db_session.add(agent)
    await db_session.flush()

    task = Task(
        title="Test task",
        description="Build a feature",
        project_id=test_project.id,
        status="in_progress",
    )
    db_session.add(task)
    await db_session.flush()

    adapter = ApiAdapter()

    with patch("huddleroom.adapters.api_adapter.memory_service_instance.search", new_callable=AsyncMock, return_value=[]):
        messages, task_ctx, k_count, ch_count = await adapter._build_messages(agent, task, db_session)

    system_msg = messages[0]["content"]
    assert "memory_write" in system_msg
    assert "memory_search" in system_msg
