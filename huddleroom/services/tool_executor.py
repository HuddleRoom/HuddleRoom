from __future__ import annotations

import json
import logging
import sys
import uuid
from collections.abc import Callable
from typing import Any
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.services.memory_service import MemoryService
from huddleroom.services.agent_response_stream import (
    AgentResponseCall,
    AgentResponseInvocation,
    extract_request_display,
)

logger = logging.getLogger(__name__)
memory_service = MemoryService()

MEMORY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "memory_write",
            "description": "Save a memory for future reference. Use for decisions, lessons learned, preferences, important context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The memory content to save"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for categorization: decision, lesson, preference, context, etc.",
                    },
                    "shared": {
                        "type": "boolean",
                        "description": "true = other agents can see this; false = private to you",
                        "default": True,
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["project", "global"],
                        "description": "project = this project only; global = across all projects",
                        "default": "project",
                    },
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "Search your memories and shared memories for relevant context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query"},
                    "limit": {"type": "integer", "description": "Max results to return", "default": 5},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter: return items matching ANY of the given tags",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_read",
            "description": "Read a specific memory item by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Memory item UUID"},
                },
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_delete",
            "description": "Delete one of your own memory items that is no longer relevant.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Memory item UUID to delete"},
                },
                "required": ["id"],
            },
        },
    },
]


MEMORY_SYSTEM_PROMPT = """You have persistent memory across sessions. Use memory_write to save decisions, lessons, and preferences. Use memory_search before complex tasks. Delete outdated items with memory_delete. Be selective—save only what future-you needs."""


def get_memory_system_prompt() -> str:
    return MEMORY_SYSTEM_PROMPT


async def execute_memory_tool(
    tool_name: str,
    arguments: dict,
    agent_id: uuid.UUID,
    project_id: uuid.UUID | None,
    db: AsyncSession,
) -> str:
    try:
        if tool_name == "memory_write":
            # Gate global scope writes
            requested_scope = arguments.get("scope", "project")
            scope = requested_scope
            if requested_scope == "global":
                from huddleroom.models.agent import Agent as AgentModel
                agent_obj = await db.get(AgentModel, agent_id)
                agent_config = (agent_obj.config or {}) if agent_obj else {}
                if not agent_config.get("allow_global_scope", False):
                    scope = "project"
                    logger.info("Agent %s attempted global scope write without allow_global_scope, downgrading to project", agent_id)

            item = await memory_service.write(
                db=db,
                agent_id=agent_id,
                project_id=project_id,
                content=arguments["content"],
                tags=arguments.get("tags", []),
                shared=arguments.get("shared", True),
                scope=scope,
            )
            return json.dumps({
                "id": str(item.id),
                "content": item.content,
                "tags": item.tags,
                "shared": item.shared,
                "scope": item.scope,
                "created_at": str(item.created_at),
            })

        elif tool_name == "memory_search":
            results = await memory_service.search(
                db=db,
                agent_id=agent_id,
                project_id=project_id,
                query=arguments["query"],
                limit=min(max(arguments.get("limit", 5), 1), 100),
                tags=arguments.get("tags"),
            )
            return json.dumps({
                "results": [
                    {
                        "id": str(r.id),
                        "content": r.content,
                        "tags": r.tags,
                        "shared": r.shared,
                        "agent_id": str(r.agent_id),
                        "scope": r.scope,
                        "relevance_score": r.relevance_score,
                        "created_at": str(r.created_at),
                    }
                    for r in results
                ],
                "count": len(results),
            })

        elif tool_name == "memory_read":
            item = await memory_service.read(
                db=db,
                agent_id=agent_id,
                project_id=project_id,
                item_id=uuid.UUID(arguments["id"]),
            )
            if item is None:
                return json.dumps({"error": "not_found", "message": "Memory item not found or not accessible"})
            return json.dumps({
                "id": str(item.id),
                "content": item.content,
                "tags": item.tags or [],
                "shared": item.shared,
                "agent_id": str(item.agent_id),
                "scope": item.scope,
                "created_at": str(item.created_at),
            })

        elif tool_name == "memory_delete":
            deleted = await memory_service.delete(
                db=db,
                agent_id=agent_id,
                item_id=uuid.UUID(arguments["id"]),
            )
            if not deleted:
                return json.dumps({"error": "not_found", "message": "Memory item not found or not owned by you"})
            return json.dumps({"deleted": True, "id": arguments["id"]})

        else:
            return json.dumps({"error": "unknown_tool", "message": f"Unknown tool: {tool_name}"})

    except Exception as e:
        logger.warning("Memory tool %s failed: %s", tool_name, e)
        return json.dumps({"error": "internal", "message": str(e)[:200]})


MAX_TOOL_ITERATIONS = 10
TOOL_BUDGET_EXHAUSTED = "Agent exhausted tool call budget (10 iterations) without producing a final response."
TOOL_EXECUTED_NO_CONTENT = "Agent executed memory tools but produced no text response."


async def run_tool_loop(
    completion_fn,
    messages: list[dict],
    tools: list[dict] | None,
    agent_id: uuid.UUID,
    project_id: uuid.UUID | None,
    db: AsyncSession,
    invocation: AgentResponseInvocation | None = None,
    response_observer: Callable[[Any], None] | None = None,
    **completion_kwargs,
) -> str:
    if tools is None:
        if invocation is None:
            resp = await completion_fn(messages=messages, **completion_kwargs)
        else:
            async with invocation.call(messages=messages) as call:
                resp = await call.complete(
                    completion_fn, {"messages": messages, **completion_kwargs}
                )
        if response_observer is not None:
            response_observer(resp)
        return resp.choices[0].message.content or ""

    messages = list(messages)  # work on a copy to avoid mutating caller's list
    last_content = None
    tools_executed = False
    previous_call: AgentResponseCall | None = None

    for iteration in range(MAX_TOOL_ITERATIONS):
        call = None
        if invocation is not None:
            call = (
                invocation.call(messages=messages)
                if previous_call is None
                else AgentResponseCall(
                    invocation,
                    extract_request_display(messages, continuation=True),
                    invocation.context.invocation_kind,
                    previous_call.call_id,
                )
            )
        if call is None:
            resp = await completion_fn(messages=messages, tools=tools, **completion_kwargs)
            call_context = None
        else:
            call_context = call

        if call_context is not None:
            await call_context.__aenter__()
        response_owner = call_context
        try:
            if call_context is not None:
                resp = await call_context.complete(
                    completion_fn,
                    {"messages": messages, "tools": tools, **completion_kwargs},
                    keep_fallback_open=True,
                )
                response_owner = call_context.response_call or call_context
            if response_observer is not None:
                response_observer(resp)
            choice = resp.choices[0]
            message = choice.message

            if message.content:
                last_content = message.content

            tool_calls = getattr(message, "tool_calls", None)
            if not tool_calls:
                content = message.content or last_content or ""
                if not content and tools_executed:
                    return TOOL_EXECUTED_NO_CONTENT
                return content

            tools_executed = True
            messages.append(message.model_dump())

            for tc in tool_calls:
                fn_name = tc.function.name
                raw_arguments = tc.function.arguments
                try:
                    fn_args = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                if response_owner is not None:
                    await response_owner.tool_started(fn_name, raw_arguments)
                result = await execute_memory_tool(
                    tool_name=fn_name,
                    arguments=fn_args,
                    agent_id=agent_id,
                    project_id=project_id,
                    db=db,
                )
                if response_owner is not None:
                    await response_owner.tool_finished(fn_name, "ok", result)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        finally:
            if response_owner is not None and response_owner is not call_context:
                await response_owner.__aexit__(*sys.exc_info())
            if call_context is not None:
                await call_context.__aexit__(*sys.exc_info())
                previous_call = response_owner

    return last_content or TOOL_BUDGET_EXHAUSTED
