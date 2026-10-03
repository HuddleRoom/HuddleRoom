import json
import uuid
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from huddleroom.services.tool_executor import (
    MEMORY_TOOLS,
    execute_memory_tool,
    run_tool_loop,
    get_memory_system_prompt,
)


@pytest.mark.asyncio
async def test_execute_memory_write(db_session, memory_agent, test_project):
    with patch("huddleroom.services.tool_executor.memory_service.write", new_callable=AsyncMock) as mock_write:
        mock_item = MagicMock()
        mock_item.id = uuid.uuid4()
        mock_item.content = "test"
        mock_item.tags = ["decision"]
        mock_item.shared = True
        mock_item.scope = "project"
        mock_item.created_at = "2026-06-01T00:00:00"
        mock_write.return_value = mock_item

        result_json = await execute_memory_tool(
            tool_name="memory_write",
            arguments={"content": "test", "tags": ["decision"]},
            agent_id=memory_agent.id,
            project_id=test_project.id,
            db=db_session,
        )

    result = json.loads(result_json)
    assert result["content"] == "test"
    assert "id" in result


@pytest.mark.asyncio
async def test_execute_unknown_tool(db_session, memory_agent, test_project):
    result_json = await execute_memory_tool(
        tool_name="unknown_tool",
        arguments={},
        agent_id=memory_agent.id,
        project_id=test_project.id,
        db=db_session,
    )
    result = json.loads(result_json)
    assert "error" in result


@pytest.mark.asyncio
async def test_run_tool_loop_no_tools():
    """Without tools, runs single completion and returns content."""
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "Hello world"
    mock_resp.choices[0].message.tool_calls = None

    mock_fn = AsyncMock(return_value=mock_resp)

    result = await run_tool_loop(
        completion_fn=mock_fn,
        messages=[{"role": "user", "content": "test"}],
        tools=None,
        agent_id=uuid.uuid4(),
        project_id=uuid.uuid4(),
        db=MagicMock(),
    )
    assert result == "Hello world"
    mock_fn.assert_called_once()


@pytest.mark.asyncio
async def test_run_tool_loop_with_tool_call():
    """With a tool call, executes tool and calls completion again."""
    # First response: tool call
    tool_call = MagicMock()
    tool_call.id = "call_123"
    tool_call.function.name = "memory_search"
    tool_call.function.arguments = '{"query": "test"}'

    first_msg = MagicMock()
    first_msg.content = None
    first_msg.tool_calls = [tool_call]
    first_msg.model_dump.return_value = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call_123", "function": {"name": "memory_search", "arguments": '{"query": "test"}'}}],
    }

    first_resp = MagicMock()
    first_resp.choices = [MagicMock()]
    first_resp.choices[0].message = first_msg

    # Second response: final content
    second_msg = MagicMock()
    second_msg.content = "Based on my memories..."
    second_msg.tool_calls = None

    second_resp = MagicMock()
    second_resp.choices = [MagicMock()]
    second_resp.choices[0].message = second_msg

    mock_fn = AsyncMock(side_effect=[first_resp, second_resp])

    with patch("huddleroom.services.tool_executor.execute_memory_tool", new_callable=AsyncMock, return_value='{"results": [], "count": 0}'):
        result = await run_tool_loop(
            completion_fn=mock_fn,
            messages=[{"role": "user", "content": "test"}],
            tools=MEMORY_TOOLS,
            agent_id=uuid.uuid4(),
            project_id=uuid.uuid4(),
            db=MagicMock(),
        )

    assert result == "Based on my memories..."
    assert mock_fn.call_count == 2
