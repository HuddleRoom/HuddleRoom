#!/usr/bin/env python3
"""Connect to a HuddleRoom project events websocket and write frames to stdout."""

from __future__ import annotations

import argparse
import asyncio
import sys
from urllib.parse import urlencode, urlparse, urlunparse

from websockets.asyncio.client import connect


DEFAULT_BASE_URL = "http://127.0.0.1:8001"


def _build_websocket_url(
    *,
    base_url: str,
    project_id: str,
    token: str | None,
    event_types: str | None,
    replay_since: str | None,
) -> str:
    parsed = urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    path = parsed.path.rstrip("/") + f"/ws/projects/{project_id}/events"

    query_params: dict[str, str] = {}
    if token:
        query_params["token"] = token
    if event_types:
        query_params["event_types"] = event_types
    if replay_since:
        query_params["replay_since"] = replay_since

    return urlunparse(
        (
            scheme,
            parsed.netloc,
            path,
            "",
            urlencode(query_params),
            "",
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Connect to a HuddleRoom project events websocket and print each frame to stdout.",
    )
    parser.add_argument("--project-id", required=True, help="Project UUID.")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"HuddleRoom base URL. Default: {DEFAULT_BASE_URL}",
    )
    parser.add_argument("--token", help="JWT or API key for websocket auth.")
    parser.add_argument(
        "--event-types",
        help="Comma-separated event types filter, passed through as-is.",
    )
    parser.add_argument(
        "--replay-since",
        help="Replay events emitted since this ISO-8601 timestamp.",
    )
    return parser.parse_args()


async def _run() -> int:
    args = _parse_args()
    websocket_url = _build_websocket_url(
        base_url=args.base_url,
        project_id=args.project_id,
        token=args.token,
        event_types=args.event_types,
        replay_since=args.replay_since,
    )

    async with connect(websocket_url) as websocket:
        async for message in websocket:
            sys.stdout.write(message)
            sys.stdout.write("\n")
            sys.stdout.flush()

    return 0


def main() -> int:
    try:
        return asyncio.run(_run())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
