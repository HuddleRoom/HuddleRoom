#!/usr/bin/env python3
"""Discover all project meetings and write their verbose websocket frames to stdout."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from urllib.parse import urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

from websockets.asyncio.client import connect


DEFAULT_BASE_URL = "http://127.0.0.1:8001"


def _build_meetings_api_url(*, base_url: str, project_id: str) -> str:
    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/") + f"/api/v1/projects/{project_id}/meetings"
    query = urlencode({"limit": "200"})
    return urlunparse((parsed.scheme, parsed.netloc, path, "", query, ""))


def _build_websocket_url(
    *,
    base_url: str,
    meeting_id: str,
    token: str | None,
    replay_since: str | None,
) -> str:
    parsed = urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    path = parsed.path.rstrip("/") + f"/ws/meetings/{meeting_id}"

    query_params: dict[str, str] = {}
    if token:
        query_params["token"] = token
    if replay_since:
        query_params["replay_since"] = replay_since

    return urlunparse((scheme, parsed.netloc, path, "", urlencode(query_params), ""))


def _discover_meeting_ids(*, base_url: str, project_id: str, token: str | None) -> list[str]:
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(_build_meetings_api_url(base_url=base_url, project_id=project_id), headers=headers)
    with urlopen(request) as response:  # noqa: S310
        meetings = json.loads(response.read().decode("utf-8"))

    return [meeting["id"] for meeting in meetings if meeting.get("id")]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover project meetings and print each verbose meeting websocket frame to stdout.",
    )
    parser.add_argument("--project-id", required=True, help="Project UUID.")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"HuddleRoom base URL. Default: {DEFAULT_BASE_URL}",
    )
    parser.add_argument("--token", help="JWT or API key for API and websocket auth.")
    parser.add_argument(
        "--replay-since",
        help="Replay events emitted since this ISO-8601 timestamp.",
    )
    return parser.parse_args()


async def _stream_meeting(*, base_url: str, meeting_id: str, token: str | None, replay_since: str | None) -> None:
    websocket_url = _build_websocket_url(
        base_url=base_url,
        meeting_id=meeting_id,
        token=token,
        replay_since=replay_since,
    )
    async with connect(websocket_url) as websocket:
        async for message in websocket:
            sys.stdout.write(f"[{meeting_id}] {message}\n")
            sys.stdout.flush()


async def _run() -> int:
    args = _parse_args()
    meeting_ids = _discover_meeting_ids(
        base_url=args.base_url,
        project_id=args.project_id,
        token=args.token,
    )
    if not meeting_ids:
        sys.stdout.write(f"No meetings found for project {args.project_id}.\n")
        sys.stdout.flush()
        return 0

    await asyncio.gather(
        *[
            _stream_meeting(
                base_url=args.base_url,
                meeting_id=meeting_id,
                token=args.token,
                replay_since=args.replay_since,
            )
            for meeting_id in meeting_ids
        ]
    )
    return 0


def main() -> int:
    try:
        return asyncio.run(_run())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
