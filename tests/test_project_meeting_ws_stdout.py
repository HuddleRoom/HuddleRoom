from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest


SCRIPT_PATH = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "project_meeting_ws_stdout.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("project_meeting_ws_stdout", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_meetings_api_url_uses_project_endpoint():
    module = _load_script_module()

    url = module._build_meetings_api_url(  # pylint: disable=protected-access
        base_url="http://127.0.0.1:8001",
        project_id="project-123",
    )

    assert url == "http://127.0.0.1:8001/api/v1/projects/project-123/meetings?limit=200"


def test_discover_meeting_ids_uses_bearer_token(monkeypatch):
    module = _load_script_module()

    captured = {}

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                [
                    {"id": "meeting-1"},
                    {"id": "meeting-2"},
                ]
            ).encode("utf-8")

    def fake_urlopen(request):
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        return _FakeResponse()

    monkeypatch.setattr(module, "urlopen", fake_urlopen)

    meeting_ids = module._discover_meeting_ids(  # pylint: disable=protected-access
        base_url="http://127.0.0.1:8001",
        project_id="project-123",
        token="secret-token",
    )

    assert meeting_ids == ["meeting-1", "meeting-2"]
    assert captured["url"] == "http://127.0.0.1:8001/api/v1/projects/project-123/meetings?limit=200"
    assert captured["authorization"] == "Bearer secret-token"


@pytest.mark.asyncio
async def test_run_streams_all_discovered_meetings(monkeypatch, capsys):
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
        "meeting-1": ['{"event_type":"meeting.turn_complete","turn_number":1}'],
        "meeting-2": ['{"event_type":"meeting.concluded"}'],
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
    assert "[meeting-1] {\"event_type\":\"meeting.turn_complete\",\"turn_number\":1}" in output_lines
    assert "[meeting-2] {\"event_type\":\"meeting.concluded\"}" in output_lines


@pytest.mark.asyncio
async def test_run_prints_message_when_project_has_no_meetings(monkeypatch, capsys):
    module = _load_script_module()

    monkeypatch.setattr(
        module,
        "_parse_args",
        lambda: SimpleNamespace(
            project_id="project-123",
            base_url="http://127.0.0.1:8001",
            token=None,
            replay_since=None,
        ),
    )
    monkeypatch.setattr(module, "_discover_meeting_ids", lambda **_: [])

    exit_code = await module._run()  # pylint: disable=protected-access

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "No meetings found for project project-123."
