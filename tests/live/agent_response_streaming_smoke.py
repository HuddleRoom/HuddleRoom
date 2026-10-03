#!/usr/bin/env python3
"""Approved verification check 5: one Orchestrator call + one agent session, live.

Bounded live smoke for the agent-response-streaming plan. Creates one project
and one API agent, connects once to the project's WebSocket, triggers exactly
one Orchestrator operation (a baseline goal-definition step) and one agent
session (an API task run), then asserts every `agent_response.*` event for
each actor is well-formed and that none of those events were persisted to
`EventLog` (they are live-stream-only).

Usage:
    .venv/bin/python tests/live/agent_response_streaming_smoke.py
    .venv/bin/python tests/live/agent_response_streaming_smoke.py --no-start-server
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

try:
    import websockets
except ImportError:
    print("ERROR: websockets not installed. Run: .venv/bin/pip install websockets")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse existing live-test env parsing, server lifecycle, and REST helpers.
from live_test import (  # noqa: E402  pylint: disable=wrong-import-position
    PROJECT_ROOT,
    HuddleRoomClient,
    check_server,
    start_server,
    stop_server,
)
from live_test_protocols import check_provider_ready  # noqa: E402  pylint: disable=wrong-import-position

sys.path.insert(0, str(PROJECT_ROOT))
from huddleroom.services.secret_redaction import redact_secrets  # noqa: E402  pylint: disable=wrong-import-position

PROJECT_NAME = "huddleroom-agent-response-streaming-smoke"
AGENT_NAME = "streaming-smoke-agent"
AGENT_PROVIDER = "openai"
AGENT_MODEL = "openai/gpt-4.1-nano"
WORKSPACE_DIR = Path(__file__).resolve().parent / "logs" / "streaming-smoke-workspace"
ORCHESTRATOR_ACTOR = ("system", "orchestrator")
RECV_TIMEOUT_SECONDS = 240


class SmokeFailure(RuntimeError):
    """Raised for any assertion or environment failure in the smoke run."""


def _redact_event(event: dict) -> dict:
    return json.loads(redact_secrets(json.dumps(event, ensure_ascii=False, default=str)))


def _diagnostics(events: list[dict], reason: str) -> str:
    lines = [f"REASON: {reason}", f"EVENTS COLLECTED: {len(events)}"]
    for event in events[-20:]:
        lines.append(json.dumps(_redact_event(event), ensure_ascii=False, sort_keys=True))
    return "\n".join(lines)


def _get_or_create_project(client: HuddleRoomClient) -> dict:
    page = client._get("/api/v1/projects", limit=200)  # pylint: disable=protected-access
    for project in page.get("items", []):
        if project["name"] == PROJECT_NAME:
            print(f"  Reusing project '{PROJECT_NAME}' ({project['id']})")
            return project
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Creating project '{PROJECT_NAME}'...")
    return client._post(  # pylint: disable=protected-access
        "/api/v1/projects",
        {
            "name": PROJECT_NAME,
            "description": "Agent response streaming smoke (check 5)",
            "workspace_path": str(WORKSPACE_DIR.resolve()),
        },
    )


def _get_or_create_agent(client: HuddleRoomClient) -> dict:
    defn = {
        "name": AGENT_NAME,
        "role": "smoke-tester",
        "provider": AGENT_PROVIDER,
        "model": AGENT_MODEL,
        "system_prompt": "You are a smoke-test agent. Reply with one short sentence.",
    }
    return client.get_or_create_agent(defn)


def _trigger_orchestrator_operation(client: HuddleRoomClient, project_id: str) -> None:
    """Create a goal and step it once through goal_definition (one Orchestrator LLM call)."""
    goal = client._post(  # pylint: disable=protected-access
        f"/api/v1/projects/{project_id}/orchestration/goals",
        {
            "objective": "Smoke: confirm agent response streaming for one Orchestrator call.",
            "original_request": "Smoke: confirm agent response streaming for one Orchestrator call.",
        },
    )
    goal_id = goal["goal"]["id"]
    client._post(  # pylint: disable=protected-access
        f"/api/v1/projects/{project_id}/orchestration/goals/{goal_id}/baseline/step",
        {"process_type": "goal_definition"},
    )


def _trigger_agent_session(client: HuddleRoomClient, project_id: str, agent_id: str) -> None:
    """Create and run a task assigned to the agent (one API agent session)."""
    task = client._post(  # pylint: disable=protected-access
        f"/api/v1/projects/{project_id}/tasks",
        {
            "title": "Streaming smoke task",
            "description": "Reply with one short sentence acknowledging this task.",
            "assigned_to": agent_id,
        },
    )
    client._post(f"/api/v1/projects/{project_id}/tasks/{task['id']}/run", {})  # pylint: disable=protected-access


async def _collect_agent_response_events(
    ws_base: str, project_id: str, target_actors: set[tuple[str, str]]
) -> list[dict]:
    uri = f"{ws_base}/ws/projects/{project_id}/events"
    events: list[dict] = []
    terminated: set[tuple[str, str]] = set()

    async with websockets.connect(uri, ping_interval=20, ping_timeout=30) as ws:
        async def _recv() -> None:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                event_type = msg.get("event_type", "")
                if not event_type.startswith("agent_response."):
                    continue
                events.append(msg)
                actor = (msg.get("actor_kind"), msg.get("actor_id"))
                print(f"    [WS] {event_type} actor={actor} operation={msg.get('operation')}")
                if event_type == "agent_response.terminal" and actor in target_actors:
                    terminated.add(actor)
                if terminated >= target_actors:
                    break

        try:
            await asyncio.wait_for(_recv(), timeout=RECV_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            raise SmokeFailure(
                f"timed out after {RECV_TIMEOUT_SECONDS}s waiting for terminal events; "
                f"terminated={sorted(terminated)} target={sorted(target_actors)}"
            ) from None

    return events


def _events_for_actor(events: list[dict], actor: tuple[str, str]) -> list[dict]:
    return [e for e in events if (e.get("actor_kind"), e.get("actor_id")) == actor]


def _assert_actor_events(actor: tuple[str, str], actor_events: list[dict]) -> None:
    if not actor_events:
        raise SmokeFailure(f"actor {actor} produced no agent_response events")

    first = actor_events[0]
    if first.get("event_type") != "agent_response.started":
        raise SmokeFailure(f"actor {actor} first event was {first.get('event_type')!r}, not started")
    if not (first.get("payload") or {}).get("request_display"):
        raise SmokeFailure(f"actor {actor} started event had no request_display")

    if not any(e.get("event_type") == "agent_response.output" for e in actor_events):
        raise SmokeFailure(f"actor {actor} had no agent_response.output event")
    if not any(e.get("event_type") == "agent_response.terminal" for e in actor_events):
        raise SmokeFailure(f"actor {actor} had no agent_response.terminal event")

    operations = {e.get("operation") for e in actor_events}
    if len(operations) != 1 or not next(iter(operations)):
        raise SmokeFailure(f"actor {actor} operation was not immutable: {operations}")

    for event in actor_events:
        if event is first:
            continue
        if "request_display" in (event.get("payload") or {}):
            raise SmokeFailure(
                f"actor {actor} non-started event {event.get('event_type')} carried request_display"
            )


def _assert_no_persisted_agent_response_events(client: HuddleRoomClient, project_id: str) -> None:
    cursor = None
    while True:
        page = client._get(  # pylint: disable=protected-access
            "/api/v1/events", project_id=project_id, cursor=cursor, limit=500
        )
        for item in page.get("items", []):
            if str(item.get("event_type", "")).startswith("agent_response."):
                raise SmokeFailure(f"EventLog row persisted a live-only event: {item.get('event_type')}")
        cursor = page.get("next_cursor")
        if not cursor:
            return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--no-start-server", action="store_true")
    parser.add_argument("--keep-server", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = f"http://localhost:{args.port}"
    ws_url = f"ws://localhost:{args.port}"

    provider_error = check_provider_ready(AGENT_PROVIDER)
    if provider_error:
        print(f"SKIP: live environment not configured: {provider_error}")
        return 0

    server_proc = None
    server_log = None
    if args.no_start_server:
        if not check_server(base_url):
            print(f"SKIP: no server running at {base_url} and --no-start-server was given")
            return 0
    elif check_server(base_url):
        print(f"Server already running at {base_url} — reusing.")
    else:
        logs_dir = Path(__file__).resolve().parent / "logs"
        logs_dir.mkdir(exist_ok=True)
        server_log_path = logs_dir / f"streaming_smoke_{time.strftime('%Y%m%d_%H%M%S')}.log"
        try:
            server_proc, server_log = start_server(base_url, args.port, PROJECT_ROOT, server_log_path)
        except Exception as exc:
            print(f"SKIP: could not start a live server: {exc}")
            return 0

    events: list[dict] = []
    try:
        client = HuddleRoomClient(base_url)
        print("\n=== Setting up fixtures ===")
        project = _get_or_create_project(client)
        agent = _get_or_create_agent(client)
        print(f"  Project: {project['id']}")
        print(f"  Agent: {agent['id']} ({agent['model']})")

        target_actors = {ORCHESTRATOR_ACTOR, ("agent", agent["id"])}

        async def _run() -> list[dict]:
            uri_project_id = project["id"]

            async def _drive_and_collect() -> list[dict]:
                collect_task = asyncio.ensure_future(
                    _collect_agent_response_events(ws_url, uri_project_id, target_actors)
                )
                # Give the WS connection a beat to register before triggering.
                await asyncio.sleep(0.5)
                print("\n=== Triggering one Orchestrator operation ===")
                await asyncio.to_thread(_trigger_orchestrator_operation, client, uri_project_id)
                print("=== Triggering one agent session ===")
                await asyncio.to_thread(_trigger_agent_session, client, uri_project_id, agent["id"])
                return await collect_task

            return await _drive_and_collect()

        events = asyncio.run(_run())

        print("\n=== Asserting per-actor event shape ===")
        for actor in sorted(target_actors):
            actor_events = _events_for_actor(events, actor)
            _assert_actor_events(actor, actor_events)
            print(f"  OK actor={actor} events={len(actor_events)}")

        print("=== Asserting no agent_response.* rows in EventLog ===")
        _assert_no_persisted_agent_response_events(client, project["id"])
        print("  OK: no persisted agent_response.* rows")

        print("\nPASS: one Orchestrator call and one agent session streamed correctly.")
        return 0

    except SmokeFailure as exc:
        print(f"\nFAIL: {exc}", file=sys.stderr)
        print(_diagnostics(events, str(exc)), file=sys.stderr)
        return 1
    except httpx.HTTPStatusError as exc:
        print(f"\nFAIL: HTTP error: {redact_secrets(str(exc))}", file=sys.stderr)
        return 1
    finally:
        if server_proc and not args.keep_server:
            stop_server(server_proc, server_log)
        elif server_log and not server_log.closed:
            server_log.close()


if __name__ == "__main__":
    sys.exit(main())
