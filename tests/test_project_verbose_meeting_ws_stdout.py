from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest


SCRIPT_PATH = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "project_verbose_meeting_ws_stdout.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("project_verbose_meeting_ws_stdout", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_run_streams_verbose_messages_for_all_discovered_meetings(monkeypatch, capsys):
    module = _load_script_module()

    monkeypatch.setattr(
        module,
        "_parse_args",
        lambda: SimpleNamespace(
            project_id="project-123",
            base_url="http://127.0.0.1:8001",
            token="secret-token",
            replay_since="1970-01-01T00:00:00+00:00",
        ),
    )
    monkeypatch.setattr(module, "_discover_meeting_ids", lambda **_: ["meeting-1", "meeting-2"])

    class _FakeWebSocket:
        def __init__(self, messages):
            self._messages = iter(messages)

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._messages)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    class _FakeConnect:
        def __init__(self, messages):
            self._messages = messages

        async def __aenter__(self):
            return _FakeWebSocket(self._messages)

        async def __aexit__(self, exc_type, exc, tb):
            return False

    opened_urls = []
    message_map = {
        "meeting-1": [
            '{"event_type":"meeting.trace","payload":{"trace":{"kind":"participant_turn","stage":"request"}}}'
        ],
        "meeting-2": [
            '{"event_type":"meeting.trace","payload":{"trace":{"kind":"moderator_select_next_speaker","stage":"response"}}}'
        ],
    }

    def fake_connect(url):
        meeting_id = url.rsplit("/", 1)[-1].split("?", 1)[0]
        opened_urls.append(url)
        return _FakeConnect(message_map[meeting_id])

    monkeypatch.setattr(module, "connect", fake_connect)

    exit_code = await module._run()  # pylint: disable=protected-access

    assert exit_code == 0
    assert len(opened_urls) == 2
    assert any("/ws/meetings/meeting-1?" in url for url in opened_urls)
    assert any("/ws/meetings/meeting-2?" in url for url in opened_urls)

    output_lines = capsys.readouterr().out.strip().splitlines()
    assert "[meeting-1] {\"event_type\":\"meeting.trace\",\"payload\":{\"trace\":{\"kind\":\"participant_turn\",\"stage\":\"request\"}}}" in output_lines
    assert "[meeting-2] {\"event_type\":\"meeting.trace\",\"payload\":{\"trace\":{\"kind\":\"moderator_select_next_speaker\",\"stage\":\"response\"}}}" in output_lines
