import io
import json
import signal
from contextlib import contextmanager

import httpx
import pytest

from tests.live import orchestration_baseline_e2e as runner


def test_packet_evidence_user_routing_and_request_trace_are_safe(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "EVENT_LOG_PATH", tmp_path / "events.jsonl")
    monkeypatch.setattr(runner, "HTTP_TRACE_PATH", tmp_path / "http-trace.jsonl")
    monkeypatch.setattr(runner, "PACKET_PATH", tmp_path / "judgment-packet.json")
    runner._REQUEST_TRACE.clear()

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/ok":
            return httpx.Response(200, json={"value": 42, "access_token": "response-secret"})
        return httpx.Response(500, json={"detail": "failed", "password": "response-secret"})

    with httpx.Client(transport=httpx.MockTransport(respond), base_url="http://rally") as client:
        assert runner._request(
            client,
            "POST",
            "/ok",
            json={"authorization": "request-secret", "value": 42},
        )["value"] == 42
        with pytest.raises(runner.Blocked):
            runner._request(client, "GET", "/failed", headers={"cookie": "request-secret"})

    decision = {
        "id": "decision-1",
        "status": "pending",
        "authority": "human",
        "decision_key": "goal_definition:adaptive:1:0:objective",
        "question": "Which outcome matters most?",
        "context": "The objective has two competing outcomes.",
        "options": ["speed", "quality"],
        "recommendation": "quality",
        "consequences": None,
        "related_gate_id": None,
        "related_action_id": None,
        "created_warning_id": None,
    }
    user_input = runner._user_input_requirement([decision], {"items": [decision], "deferred_count": 0})
    process = {
        "id": "process-1",
        "process_type": "goal_definition",
        "status": "waiting_decision",
        "run_id": "run-1",
        "inputs": {"api_key": "input-secret"},
        "outputs": {"value": 42, "token": "output-secret"},
    }
    packet = runner._packet(
        "project-1",
        "goal-1",
        process,
        [],
        [],
        created_at="2026-01-01T00:00:00Z",
        terminal_slice=["authorization=terminal-secret", "phase finished"],
        request_trace=runner._REQUEST_TRACE,
        user_input=user_input,
    )
    changed_observations = runner._packet(
        "project-1",
        "goal-1",
        process,
        [],
        [],
        created_at="2026-01-01T00:00:00Z",
        terminal_slice=["different log"],
        request_trace=[],
        user_input=user_input,
    )

    assert packet["evidence_fingerprint"] == changed_observations["evidence_fingerprint"]
    assert packet["packet_fingerprint"] != changed_observations["packet_fingerprint"]
    assert packet["inputs"]["api_key"] == runner.REDACTED
    assert packet["outputs"]["token"] == runner.REDACTED
    assert packet["observations"]["terminal_slice"] == [
        "authorization=[REDACTED]",
        "phase finished",
    ]
    assert packet["observations"]["request_trace"][0]["request"]["json"]["authorization"] == runner.REDACTED
    assert packet["observations"]["request_trace"][0]["response"]["body"]["access_token"] == runner.REDACTED
    assert packet["observations"]["request_trace"][1]["response"]["body"]["password"] == runner.REDACTED
    assert packet["phase"]["purpose"]
    assert len(packet["phase"]["acceptance_criteria"]) >= 3
    assert user_input["items"][0]["delegation_allowed"] is True
    assert runner._delegated_answer_request(packet, "decision-1:quality") == (
        "/api/v1/projects/project-1/orchestration/goals/goal-1/decisions/decision-1/answer",
        {"selected_option": "quality", "reason": "Explicitly delegated baseline clarification"},
    )
    with pytest.raises(runner.Blocked):
        runner._delegated_answer_request(
            {**packet, "user_input": {"items": [{**user_input["items"][0], "delegation_allowed": False}]}},
            "decision-1:quality",
        )

    assert runner._emit_user_input(packet) == runner.HUMAN_JUDGMENT_REQUIRED
    assert "USER INPUT REQUIRED" in capsys.readouterr().out
    trace_lines = [json.loads(line) for line in (tmp_path / "http-trace.jsonl").read_text().splitlines()]
    assert len(trace_lines) == 2


def test_reusable_verdict_requires_exact_process_and_fingerprint_with_pass_or_minor():
    passed = {"process_id": "process-1", "fingerprint": "fingerprint-1", "verdict": "pass"}
    minor = {"process_id": "process-2", "fingerprint": "fingerprint-2", "verdict": "minor"}
    human = {"process_id": "process-3", "fingerprint": "fingerprint-3", "verdict": "needs_human_judgment"}
    records = [passed, minor, human]

    assert runner.reusable_verdict(records, "process-1", "fingerprint-1") == passed
    assert runner.reusable_verdict(records, "process-2", "fingerprint-2") == minor
    assert runner.reusable_verdict(records, "process-1", "fingerprint-2") is None
    assert runner.reusable_verdict(records, "process-3", "fingerprint-3") is None


def test_controlled_server_always_stops_child_and_captures_terminal(monkeypatch):
    launched = {}

    class Process:
        pid = 123
        stdout = io.StringIO("authorization=server-secret\nserver ready\n")
        stderr = io.StringIO("warning\n")
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            launched["terminated"] = True
            self.returncode = 0

        def wait(self, timeout=None):
            launched["wait_timeout"] = timeout
            return self.returncode

        def kill(self):
            launched["killed"] = True

    def popen(command, **kwargs):
        launched.update(command=command, kwargs=kwargs)
        return Process()

    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    monkeypatch.setattr(runner.os, "killpg", lambda pid, sig: launched.update(killpg=(pid, sig)))
    monkeypatch.setattr(runner, "_wait_until_ready", lambda *_: None)
    runner._TERMINAL_LINES.clear()

    with pytest.raises(RuntimeError):
        with runner._controlled_server() as base_url:
            assert base_url.startswith("http://127.0.0.1:")
            raise RuntimeError("body failed")

    assert launched["command"][:9] == [
        "onecli",
        "run",
        "--agent",
        "rally-onecli",
        "--",
        runner.sys.executable,
        "-m",
        "huddleroom.cli",
        "serve",
    ]
    assert launched["command"][9:] == [
        "--reload",
        "--host",
        "127.0.0.1",
        "--port",
        base_url.rsplit(":", 1)[1],
    ]
    assert launched["kwargs"]["start_new_session"] is True
    assert launched["kwargs"]["env"]["HUDDLEROOM_DEBUG"] == "true"
    assert launched["killpg"] == (123, signal.SIGTERM)
    assert launched["wait_timeout"] == 5
    assert runner._TERMINAL_LINES == [
        "stdout: authorization=[REDACTED]",
        "stdout: server ready",
        "stderr: warning",
    ]


def test_waiting_process_without_decisions_advances_through_debug_step(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "CHECKPOINT_PATH", tmp_path / "checkpoint.json")
    monkeypatch.setattr(runner, "EVENT_LOG_PATH", tmp_path / "events.jsonl")
    monkeypatch.setattr(runner, "HTTP_TRACE_PATH", tmp_path / "http-trace.jsonl")
    monkeypatch.setattr(runner, "VERDICT_LOG_PATH", tmp_path / "verdicts.jsonl")
    monkeypatch.setattr(runner, "PACKET_PATH", tmp_path / "judgment-packet.json")
    monkeypatch.setattr(runner, "ARTIFACT_DIR", tmp_path)
    runner._REQUEST_TRACE.clear()
    process = {"id": "process-1", "process_type": "goal_definition", "status": "waiting_decision", "outputs": {}}
    stepped = []

    def respond(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/processes"):
            return httpx.Response(200, json=[{**process, "status": "completed" if stepped else "waiting_decision"}])
        if request.method == "GET" and request.url.path.endswith("/decisions"):
            return httpx.Response(200, json=[])
        if request.method == "GET" and request.url.path.endswith("/decisions/checkpoint"):
            return httpx.Response(200, json={"items": [], "deferred_count": 0})
        if request.method == "POST" and request.url.path.endswith("/debug/baseline/step"):
            stepped.append(request.url.path)
            return httpx.Response(200, json={"process": {**process, "status": "completed"}})
        if request.method == "GET" and request.url.path.endswith("/warnings"):
            return httpx.Response(200, json=[])
        raise AssertionError(f"unexpected request: {request.method} {request.url.path}")

    with httpx.Client(transport=httpx.MockTransport(respond), base_url="http://rally") as client:
        assert runner._step(client, "project-1", "goal-1", 0, 0) == runner.JUDGMENT_REQUIRED

    assert stepped == ["/api/v1/projects/project-1/orchestration/goals/goal-1/debug/baseline/step"]


def test_blocker_exit_persists_fresh_redacted_packet_and_server_log(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "CHECKPOINT_PATH", tmp_path / "checkpoint.json")
    monkeypatch.setattr(runner, "EVENT_LOG_PATH", tmp_path / "events.jsonl")
    monkeypatch.setattr(runner, "HTTP_TRACE_PATH", tmp_path / "http-trace.jsonl")
    monkeypatch.setattr(runner, "PACKET_PATH", tmp_path / "judgment-packet.json")
    monkeypatch.setattr(runner, "VERDICT_LOG_PATH", tmp_path / "verdicts.jsonl")
    monkeypatch.setattr(runner, "ARTIFACT_DIR", tmp_path)

    @contextmanager
    def controlled_server():
        runner._TERMINAL_LINES.append("stderr: provider tail authorization=server-secret")
        yield "http://rally"

    monkeypatch.setattr(runner, "_controlled_server", controlled_server)
    monkeypatch.setattr(runner, "_resolve_target", lambda _client: ("project-1", "goal-1"))
    monkeypatch.setattr(runner, "_step", lambda *_args: runner._block("final provider failure"))

    assert runner.run(["--step", "--controlled-server"]) == 1

    packet_text = (tmp_path / "judgment-packet.json").read_text()
    server_log = (tmp_path / "server.log").read_text()
    assert "provider tail" in packet_text
    assert "provider tail" in server_log
    assert "server-secret" not in packet_text
    assert "server-secret" not in server_log


def test_success_exit_persists_server_log_after_controlled_server_shutdown(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "CHECKPOINT_PATH", tmp_path / "checkpoint.json")
    monkeypatch.setattr(runner, "EVENT_LOG_PATH", tmp_path / "events.jsonl")
    monkeypatch.setattr(runner, "HTTP_TRACE_PATH", tmp_path / "http-trace.jsonl")
    monkeypatch.setattr(runner, "PACKET_PATH", tmp_path / "judgment-packet.json")
    monkeypatch.setattr(runner, "VERDICT_LOG_PATH", tmp_path / "verdicts.jsonl")
    monkeypatch.setattr(runner, "ARTIFACT_DIR", tmp_path)

    @contextmanager
    def controlled_server():
        yield "http://rally"
        runner._TERMINAL_LINES.append("stdout: final success token=server-secret")

    monkeypatch.setattr(runner, "_controlled_server", controlled_server)
    monkeypatch.setattr(runner, "_resolve_target", lambda _client: ("project-1", "goal-1"))
    monkeypatch.setattr(runner, "_step", lambda *_args: 0)

    assert runner.run(["--step", "--controlled-server"]) == 0

    server_log = (tmp_path / "server.log").read_text()
    assert "final success" in server_log
    assert "server-secret" not in server_log


def test_resolve_target_discards_422_cached_target_and_discovers_active_goal(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "CHECKPOINT_PATH", tmp_path / "checkpoint.json")
    monkeypatch.setattr(runner, "EVENT_LOG_PATH", tmp_path / "events.jsonl")
    monkeypatch.setattr(runner, "HTTP_TRACE_PATH", tmp_path / "http-trace.jsonl")
    monkeypatch.setattr(runner, "PACKET_PATH", tmp_path / "judgment-packet.json")
    monkeypatch.setattr(runner, "VERDICT_LOG_PATH", tmp_path / "verdicts.jsonl")
    monkeypatch.setattr(runner, "ARTIFACT_DIR", tmp_path)
    runner.CHECKPOINT_PATH.write_text(json.dumps({"project_id": "stale-project", "goal_id": "stale-goal"}))

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/projects/stale-project":
            return httpx.Response(422, json={"detail": "malformed cached project"})
        if request.url.path == "/api/v1/projects":
            return httpx.Response(200, json={"items": [{"id": "project-1", "name": runner.PROJECT_NAME}]})
        if request.url.path == "/api/v1/projects/project-1/orchestration/goals":
            return httpx.Response(200, json={"items": [{"id": "goal-1", "status": "active"}]})
        raise AssertionError(f"unexpected request: {request.method} {request.url.path}")

    with httpx.Client(transport=httpx.MockTransport(respond), base_url="http://rally") as client:
        assert runner._resolve_target(client) == ("project-1", "goal-1")


def test_resolve_target_accepts_cached_blocked_goal(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "CHECKPOINT_PATH", tmp_path / "checkpoint.json")
    runner.CHECKPOINT_PATH.write_text(json.dumps({"project_id": "project-1", "goal_id": "goal-1"}))

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/projects/project-1":
            return httpx.Response(200, json={"id": "project-1", "name": runner.PROJECT_NAME})
        if request.url.path.endswith("/orchestration/goals/goal-1"):
            return httpx.Response(200, json={"goal": {
                "id": "goal-1", "project_id": "project-1", "status": "blocked"
            }})
        raise AssertionError(f"unexpected request: {request.method} {request.url.path}")

    with httpx.Client(transport=httpx.MockTransport(respond), base_url="http://rally") as client:
        assert runner._resolve_target(client) == ("project-1", "goal-1")


def test_invocation_evidence_retains_full_redacted_artifacts_and_state(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(runner, "PACKET_PATH", tmp_path / "judgment-packet.json")
    monkeypatch.setattr(runner, "_INVOCATION_PACKET", {"token": runner.REDACTED, "value": 1}, raising=False)
    runner._TERMINAL_LINES[:] = ["stderr: full provider error token=server-secret"]
    runner._REQUEST_TRACE[:] = [{"path": "/example", "response": {"body": {"token": "trace-secret"}}}]
    runner.PACKET_PATH.write_text(json.dumps({"token": "packet-secret", "value": 1}))
    states = {
        "before": {"goal": {"status": "active"}, "processes": [], "decisions": []},
        "after": {"goal": {"status": "blocked"}, "processes": [], "decisions": []},
    }

    invocation_dir = runner._persist_invocation_evidence("invocation-1", states)

    assert invocation_dir == tmp_path / "invocations" / "invocation-1"
    assert "full provider error" in (invocation_dir / "server.log").read_text()
    assert "server-secret" not in (invocation_dir / "server.log").read_text()
    assert "trace-secret" not in (invocation_dir / "http-trace.jsonl").read_text()
    assert "packet-secret" not in (invocation_dir / "judgment-packet.json").read_text()
    assert json.loads((invocation_dir / "state-before.json").read_text())["goal"]["status"] == "active"
    assert json.loads((invocation_dir / "state-after.json").read_text())["goal"]["status"] == "blocked"


def test_redact_scrubs_secret_patterns_from_nested_string_leaves():
    redacted = runner._redact({
        "message": "provider sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ012345 retained-tail",
        "nested": [{"detail": "failure rk-abcdefghijklmnopqrstuvwxyz012345 useful-tail"}],
        "short": "sk-short",
    })

    assert redacted["message"] == "provider [REDACTED] retained-tail"
    assert redacted["nested"][0]["detail"] == "failure [REDACTED] useful-tail"
    assert redacted["short"] == "sk-short"


def test_persisted_state_redacts_string_leaf_but_keeps_diagnostic_tail(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(runner, "PACKET_PATH", tmp_path / "judgment-packet.json")
    runner._TERMINAL_LINES.clear()
    runner._REQUEST_TRACE.clear()

    invocation_dir = runner._persist_invocation_evidence("invocation-state", {
        "before": {},
        "after": {"goal": {"body": {
            "detail": "failure nvapi-ABCDEFGHIJKLMNOPQRSTUVWXYZ012345 diagnostic-tail"
        }}},
    })

    state = (invocation_dir / "state-after.json").read_text()
    assert "diagnostic-tail" in state
    assert "nvapi-ABCDEFGHIJKLMNOPQRSTUVWXYZ012345" not in state


def test_no_packet_invocation_does_not_copy_stale_canonical_packet(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(runner, "PACKET_PATH", tmp_path / "judgment-packet.json")
    runner.PACKET_PATH.write_text(json.dumps({"value": "stale"}))
    runner._TERMINAL_LINES.clear()
    runner._REQUEST_TRACE.clear()
    monkeypatch.setattr(runner, "_INVOCATION_PACKET", None, raising=False)

    invocation_dir = runner._persist_invocation_evidence("invocation-no-packet", {})

    assert not (invocation_dir / "judgment-packet.json").exists()
    assert json.loads(runner.PACKET_PATH.read_text()) == {"value": "stale"}


def test_invocation_persists_emitted_packet_not_later_canonical_contents(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(runner, "PACKET_PATH", tmp_path / "judgment-packet.json")
    runner._TERMINAL_LINES.clear()
    runner._REQUEST_TRACE.clear()
    monkeypatch.setattr(runner, "_INVOCATION_PACKET", None, raising=False)
    runner._write_diagnostic_packet("packet A")
    runner.PACKET_PATH.write_text(json.dumps({"reason": "packet B"}))

    invocation_dir = runner._persist_invocation_evidence("invocation-race", {})

    assert json.loads((invocation_dir / "judgment-packet.json").read_text())["reason"] == "packet A"
    assert json.loads(runner.PACKET_PATH.read_text())["reason"] == "packet B"


def test_missing_huddleroom_url_persists_blocker_invocation_evidence(tmp_path, monkeypatch):
    monkeypatch.delenv("HUDDLEROOM_URL", raising=False)
    monkeypatch.delenv("RALLY_URL", raising=False)
    monkeypatch.setattr(runner, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(runner, "CHECKPOINT_PATH", tmp_path / "checkpoint.json")
    monkeypatch.setattr(runner, "EVENT_LOG_PATH", tmp_path / "events.jsonl")
    monkeypatch.setattr(runner, "HTTP_TRACE_PATH", tmp_path / "http-trace.jsonl")
    monkeypatch.setattr(runner, "PACKET_PATH", tmp_path / "judgment-packet.json")
    monkeypatch.setattr(runner, "VERDICT_LOG_PATH", tmp_path / "verdicts.jsonl")

    assert runner.run(["--step"]) == 1

    invocation_dirs = list((tmp_path / "invocations").iterdir())
    assert len(invocation_dirs) == 1
    invocation_dir = invocation_dirs[0]
    assert json.loads((invocation_dir / "state-before.json").read_text()) == {}
    assert json.loads((invocation_dir / "state-after.json").read_text()) == {}
    assert (invocation_dir / "server.log").read_text() == "\n"
    assert (invocation_dir / "http-trace.jsonl").read_text() == ""
    assert json.loads((invocation_dir / "judgment-packet.json").read_text())["reason"] == "HUDDLEROOM_URL is required"
