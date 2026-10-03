"""One-step live runner for agent-judged orchestration baseline output."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from typing import Any, NoReturn
import uuid

import httpx


PROCESS_ORDER = (
    "goal_definition",
    "manager_selection",
    "agent_definition_review",
    "team_hierarchy",
    "effectiveness_review",
    "goal_closeout",
)
# Types whose waiting_decision is a judgeable output (a recommendation/rationale to
# review), not a clarification requiring user input. Explicit allowlist so the original
# 4 phases' waiting_decision -> user-input routing is untouched.
JUDGEABLE_WAITING_TYPES = frozenset({"effectiveness_review", "goal_closeout"})

# goal_closeout's advance_process() raises 409 for several distinct reasons. Only these
# exact "goal has not executed far enough yet" messages are a clean skip. Every other 409
# -- active blocker/hard_stop or unacknowledged warnings, budget exceeded, an invalid
# final-summary payload, a missing session, a blocked/untickable run -- is a real
# condition that must still hard-block for review, so it is deliberately NOT listed here.
_CLOSEOUT_NOT_READY_409_DETAILS = frozenset({
    "goal_definition, manager_selection, agent_definition_review, and team_hierarchy "
    "must all be terminal before goal_closeout can advance",
    "All orchestration gates must be accepted",
    "Accepted gate evidence is missing",
    "Final summary evidence is missing",
})
TERMINAL_STATUSES = {"completed", "skipped"}
REUSABLE_VERDICTS = {"pass", "minor"}
VERDICTS = ("pass", "minor", "needs_human_judgment")
PROJECT_NAME = os.environ.get("HUDDLEROOM_BASELINE_PROJECT_NAME", "HuddleRoom Orchestration Baseline")
ARTIFACT_DIR = Path(__file__).resolve().parent / "logs" / "orchestration-baseline-e2e"
CHECKPOINT_PATH = ARTIFACT_DIR / "checkpoint.json"
EVENT_LOG_PATH = ARTIFACT_DIR / "events.jsonl"
PACKET_PATH = ARTIFACT_DIR / "judgment-packet.json"
VERDICT_LOG_PATH = ARTIFACT_DIR / "verdicts.jsonl"
HTTP_TRACE_PATH = ARTIFACT_DIR / "http-trace.jsonl"
JUDGMENT_REQUIRED = 3
HUMAN_JUDGMENT_REQUIRED = 4
REDACTED = "[REDACTED]"
_SECRET_WORDS = {
    "auth",
    "authorization",
    "cookie",
    "credential",
    "credentials",
    "jwt",
    "password",
    "passwd",
    "secret",
    "token",
}
_SECRET_COMPOUNDS = {
    "accesstoken",
    "apikey",
    "clientsecret",
    "encryptionkey",
    "privatekey",
    "refreshtoken",
    "signingkey",
}
_ACTIVE_TARGET: dict[str, str] = {}
_REQUEST_TRACE: list[dict] = []
_TERMINAL_LINES: list[str] = []
_INVOCATION_PACKET: dict | None = None
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASES = {
    "goal_definition": {
        "purpose": "Turn the request into an actionable goal with explicit success conditions.",
        "acceptance_criteria": [
            "The objective and success criteria are concrete enough to guide execution.",
            "Constraints, budget, assumptions, and accepted clarifications are represented without contradiction.",
            "Unsafe ambiguity is surfaced for user input rather than silently accepted.",
        ],
    },
    "manager_selection": {
        "purpose": "Choose accountable management and an authority model appropriate to the goal.",
        "acceptance_criteria": [
            "The selected manager, or explicit no-manager choice, fits the goal and available candidates.",
            "The authority model and rationale are internally consistent.",
            "Candidate evidence and required gates are complete enough to justify the selection.",
        ],
    },
    "agent_definition_review": {
        "purpose": "Verify that the agents needed for the goal have usable, current definitions.",
        "acceptance_criteria": [
            "Reviewed agents cover the work the goal requires and ineligible definitions are identified.",
            "Review and coverage fingerprints describe the current target set.",
            "Warnings, gates, and review evidence agree with the reported eligibility.",
        ],
    },
    "team_hierarchy": {
        "purpose": "Establish a coherent reporting and coordination structure for execution.",
        "acceptance_criteria": [
            "Required work functions map to agents with no unexplained gaps.",
            "The hierarchy has no dangling or contradictory authority relationships.",
            "Independent verification is possible or any override is explicitly evidenced.",
        ],
    },
    "effectiveness_review": {
        "purpose": "Judge whether ongoing execution should continue, revise, split, or pause based on triggered evidence.",
        "acceptance_criteria": [
            "The trigger(s) that fired are evidenced and not fabricated.",
            "The recommended disposition follows from the shown checks and triggers.",
            "A failed check never yields a 'continue' recommendation.",
        ],
    },
    "goal_closeout": {
        "purpose": "Authorize and record goal completion from accepted gates and evidence.",
        "acceptance_criteria": [
            "Declared success criteria are matched against real accepted evidence, not assumed.",
            "The closeout rationale/manifest is internally consistent with gates and warnings.",
            "Sign-off authority and consequences are correctly attributed.",
        ],
    },
}


def _secret_key(key: object) -> bool:
    text = str(key)
    words = set(re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text).lower().replace("-", "_").split("_"))
    compact = re.sub(r"[^a-z0-9]", "", text.lower())
    return bool(words & _SECRET_WORDS) or any(compound in compact for compound in _SECRET_COMPOUNDS)


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): REDACTED if _secret_key(key) else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    return value


def _redact_text(value: str) -> str:
    value = re.sub(r"\b(?:sk|rk|pk|nvapi)-[A-Za-z0-9_-]{16,}\b", REDACTED, value)
    value = re.sub(r"(?i)(authorization|cookie|credentials?|jwt|password|passwd|secret|token)(\s*[=:]\s*)(?:bearer\s+)?[^\s,;]+", rf"\1\2{REDACTED}", value)
    value = re.sub(r"(?i)bearer\s+[^\s,;]+", f"Bearer {REDACTED}", value)
    return re.sub(r"(://)[^/\s:@]+:[^@\s/]+@", rf"\1{REDACTED}@", value)


def canonical_fingerprint(output: Any) -> str:
    """Hash canonical JSON after recursively removing secret values."""
    canonical = json.dumps(
        _redact(output),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def reusable_verdict(
    records: list[dict], process_id: str, fingerprint: str
) -> dict | None:
    """Return the latest exact reusable judgment, if any."""
    for record in reversed(records):
        if record.get("process_id") == process_id and record.get("fingerprint") == fingerprint:
            return record if record.get("verdict") in REUSABLE_VERDICTS else None
    return None


def current_processes(processes: list[dict]) -> dict[str, dict]:
    current: dict[str, dict] = {}
    for process in processes:
        if process.get("superseded_by_id") is not None:
            continue
        process_type = process.get("process_type")
        if not isinstance(process_type, str) or not process_type or process_type in current:
            _block("invalid current process list", process=_safe_process(process))
        current[process_type] = process
    return current


def blocker_reason(process: dict) -> str | None:
    if process.get("status") == "waiting_decision":
        return "waiting for approval"
    outputs = process.get("outputs")
    if process.get("clarification_limit_reached") or (
        isinstance(outputs, dict) and outputs.get("clarification_limit_reached")
    ):
        return "clarification limit reached"
    return None


@dataclass
class Checkpoint:
    project_id: str
    goal_id: str
    run_id: str | None = None
    last_step_id: str | None = None
    last_step_status: str | None = None
    updated_at: str | None = None


class Blocked(RuntimeError):
    """A live-run condition that requires human attention."""


def _capture_stream(stream: Any, label: str) -> None:
    for line in stream:
        _TERMINAL_LINES.append(f"{label}: {_redact_text(line.rstrip())}")


def _wait_until_ready(process: subprocess.Popen, base_url: str) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            _block("controlled HuddleRoom server exited before readiness", returncode=process.returncode)
        try:
            response = httpx.get(f"{base_url}/health", timeout=0.5)
        except httpx.HTTPError as exc:
            _trace_request("GET", "/health", {}, {"error": type(exc).__name__, "readiness": True})
        else:
            _trace_request("GET", "/health", {}, {**_response_trace(response), "readiness": True})
            if response.status_code == 200:
                return
        time.sleep(0.1)
    _block("controlled HuddleRoom server readiness timed out")


@contextmanager
def _controlled_server():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env["HUDDLEROOM_DEBUG"] = "true"
    env["HUDDLEROOM_ORCHESTRATION_BASELINE_E2E"] = "true"
    process = subprocess.Popen(
        [
            "onecli",
            "run",
            "--agent",
            "rally-onecli",
            "--",
            sys.executable,
            "-m",
            "huddleroom.cli",
            "serve",
            "--reload",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=_PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    threads = [
        threading.Thread(target=_capture_stream, args=(process.stdout, "stdout"), daemon=True),
        threading.Thread(target=_capture_stream, args=(process.stderr, "stderr"), daemon=True),
    ]
    for thread in threads:
        thread.start()
    try:
        _wait_until_ready(process, base_url)
        yield base_url
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for thread in threads:
            thread.join(timeout=1)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_redact(record), ensure_ascii=False, sort_keys=True) + "\n")


def _append_event(kind: str, **details: Any) -> None:
    _append_jsonl(EVENT_LOG_PATH, {"at": _now(), "kind": kind, **details})


def _trace_request(method: str, path: str, request: dict, response: dict) -> None:
    record = _redact({"at": _now(), "method": method, "path": path, "request": request, "response": response})
    _REQUEST_TRACE.append(record)
    _append_jsonl(HTTP_TRACE_PATH, record)


def _response_trace(response: httpx.Response) -> dict:
    try:
        body = response.json()
    except ValueError:
        body = "[NON_JSON_BODY_OMITTED]"
    return {"status": response.status_code, "body": body}


def _perform_request(client: httpx.Client, method: str, path: str, **kwargs: Any) -> httpx.Response:
    request = {key: kwargs[key] for key in ("params", "json", "data", "headers") if key in kwargs}
    try:
        response = client.request(method, path, **kwargs)
    except httpx.HTTPError as exc:
        _trace_request(method, path, request, {"error": type(exc).__name__})
        _block("HTTP request failed", method=method, path=path, error=type(exc).__name__)
    _trace_request(method, path, request, _response_trace(response))
    return response


def _block(reason: str, **details: Any) -> NoReturn:
    _append_event("blocker", reason=reason, **_ACTIVE_TARGET, **details)
    raise Blocked(reason)


def _write_diagnostic_packet(reason: str) -> None:
    global _INVOCATION_PACKET

    packet = _redact(
        {
            "kind": "blocker",
            "reason": reason,
            "target": _ACTIVE_TARGET,
            "observations": {
                "terminal_slice": [_redact_text(line) for line in _TERMINAL_LINES],
                "request_trace": _REQUEST_TRACE,
            },
        }
    )
    PACKET_PATH.parent.mkdir(parents=True, exist_ok=True)
    PACKET_PATH.write_text(json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _INVOCATION_PACKET = packet


def _write_server_log() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "server.log").write_text(
        "\n".join(_redact_text(line) for line in _TERMINAL_LINES) + "\n", encoding="utf-8"
    )


def _capture_state(client: httpx.Client, project_id: str, goal_id: str) -> dict:
    base = f"/api/v1/projects/{project_id}/orchestration/goals/{goal_id}"
    state = {}
    for key, path in {
        "goal": base,
        "processes": f"{base}/processes",
        "decisions": f"{base}/decisions",
    }.items():
        try:
            response = client.get(path)
            trace = _response_trace(response)
            _trace_request("GET", path, {}, trace)
            state[key] = trace
        except httpx.HTTPError as exc:
            state[key] = {"error": type(exc).__name__}
    return _redact(state)


def _persist_invocation_evidence(invocation_id: str, states: dict) -> Path:
    invocation_dir = ARTIFACT_DIR / "invocations" / invocation_id
    invocation_dir.mkdir(parents=True, exist_ok=False)
    (invocation_dir / "server.log").write_text(
        "\n".join(_redact_text(line) for line in _TERMINAL_LINES) + "\n", encoding="utf-8"
    )
    (invocation_dir / "http-trace.jsonl").write_text(
        "".join(json.dumps(_redact(record), ensure_ascii=False, sort_keys=True) + "\n" for record in _REQUEST_TRACE),
        encoding="utf-8",
    )
    for label in ("before", "after"):
        (invocation_dir / f"state-{label}.json").write_text(
            json.dumps(_redact(states.get(label, {})), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if _INVOCATION_PACKET is not None:
        (invocation_dir / "judgment-packet.json").write_text(
            json.dumps(_INVOCATION_PACKET, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return invocation_dir


def _safe_process(process: dict | None) -> dict:
    if not process:
        return {}
    return {
        "id": process.get("id"),
        "process_type": process.get("process_type"),
        "status": process.get("status"),
        "run_id": process.get("run_id"),
    }


def _write_checkpoint(project_id: str, goal_id: str, process: dict | None = None) -> None:
    checkpoint = Checkpoint(
        project_id=project_id,
        goal_id=goal_id,
        run_id=str(process.get("run_id")) if process and process.get("run_id") else None,
        last_step_id=str(process.get("id")) if process and process.get("id") else None,
        last_step_status=process.get("status") if process else None,
        updated_at=_now(),
    )
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(json.dumps(asdict(checkpoint), sort_keys=True), encoding="utf-8")


def _read_checkpoint() -> Checkpoint | None:
    try:
        payload = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        checkpoint = Checkpoint(project_id=payload["project_id"], goal_id=payload["goal_id"])
    except (FileNotFoundError, TypeError, ValueError, KeyError):
        return None
    if not all(isinstance(value, str) and value for value in (checkpoint.project_id, checkpoint.goal_id)):
        return None
    return checkpoint


def _request(client: httpx.Client, method: str, path: str, *, ignore_response=None, **kwargs: Any) -> Any:
    response = _perform_request(client, method, path, **kwargs)
    if response.status_code in {401, 403}:
        _block("authentication failed", method=method, path=path, status=response.status_code)
    if response.status_code == 404 and path.endswith(("/debug/baseline/step", "/debug/baseline/rerun-last")):
        _block("orchestration debug is disabled", method=method, path=path, status=404)
    if ignore_response is not None and ignore_response(response):
        return None
    if response.is_error:
        _block("HTTP request failed", method=method, path=path, status=response.status_code)
    try:
        return response.json()
    except ValueError:
        _block("invalid JSON response", method=method, path=path)


def _closeout_not_ready(response: httpx.Response) -> bool:
    if response.status_code != 409:
        return False
    try:
        detail = response.json().get("detail")
    except ValueError:
        return False
    return isinstance(detail, str) and detail in _CLOSEOUT_NOT_READY_409_DETAILS


def _cached_get(client: httpx.Client, path: str) -> dict | None:
    response = _perform_request(client, "GET", path)
    if response.status_code in {401, 403}:
        _block("authentication failed", method="GET", path=path, status=response.status_code)
    if response.status_code in {404, 422}:
        return None
    if response.is_error:
        _block("HTTP request failed", method="GET", path=path, status=response.status_code)
    try:
        payload = response.json()
    except ValueError:
        _block("invalid JSON response", method="GET", path=path)
    return payload if isinstance(payload, dict) else None


def _page(client: httpx.Client, path: str, **params: Any) -> list[dict]:
    items: list[dict] = []
    cursor: str | None = None
    while True:
        payload = _request(client, "GET", path, params={**params, "cursor": cursor, "limit": 100})
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            _block("invalid paginated response", path=path)
        items.extend(item for item in payload["items"] if isinstance(item, dict))
        cursor = payload.get("next_cursor")
        if not cursor:
            return items


def _valid_cached_target(client: httpx.Client, checkpoint: Checkpoint) -> bool:
    project = _cached_get(client, f"/api/v1/projects/{checkpoint.project_id}")
    if project is None or project.get("name") != PROJECT_NAME:
        return False
    detail = _cached_get(
        client,
        f"/api/v1/projects/{checkpoint.project_id}/orchestration/goals/{checkpoint.goal_id}",
    )
    goal = detail.get("goal") if detail else None
    return (
        isinstance(goal, dict)
        and str(goal.get("id")) == checkpoint.goal_id
        and str(goal.get("project_id")) == checkpoint.project_id
    )


def _discover_target(client: httpx.Client) -> tuple[str, str]:
    projects = [project for project in _page(client, "/api/v1/projects") if project.get("name") == PROJECT_NAME]
    if len(projects) != 1 or not isinstance(projects[0].get("id"), str):
        _block("target discovery failed", target="project", matches=len(projects))
    project_id = projects[0]["id"]
    goals = [
        goal
        for goal in _page(client, f"/api/v1/projects/{project_id}/orchestration/goals", status="active")
        if goal.get("status") == "active"
    ]
    if len(goals) != 1 or not isinstance(goals[0].get("id"), str):
        _block("target discovery failed", target="active goal", matches=len(goals))
    return project_id, goals[0]["id"]


def _resolve_target(client: httpx.Client) -> tuple[str, str]:
    checkpoint = _read_checkpoint()
    if checkpoint is not None and _valid_cached_target(client, checkpoint):
        return checkpoint.project_id, checkpoint.goal_id
    return _discover_target(client)


def _processes(client: httpx.Client, project_id: str, goal_id: str) -> list[dict]:
    payload = _request(
        client,
        "GET",
        f"/api/v1/projects/{project_id}/orchestration/goals/{goal_id}/processes",
    )
    if not isinstance(payload, list) or not all(isinstance(process, dict) for process in payload):
        _block("invalid process list")
    return payload


def _warnings(client: httpx.Client, project_id: str, goal_id: str, process: dict) -> list[dict]:
    payload = _request(
        client,
        "GET",
        f"/api/v1/projects/{project_id}/orchestration/goals/{goal_id}/warnings",
    )
    if not isinstance(payload, list) or not all(isinstance(warning, dict) for warning in payload):
        _block("invalid warning list")
    outputs = process.get("outputs")
    raw_warning_ids = outputs.get("warning_ids") or [] if isinstance(outputs, dict) else []
    if not isinstance(raw_warning_ids, list) or not all(isinstance(item, str) for item in raw_warning_ids):
        _block("invalid process output", process=_safe_process(process))
    warning_ids = set(raw_warning_ids)
    missing_warning_ids = warning_ids - {warning.get("id") for warning in payload}
    if missing_warning_ids:
        _block(
            "missing linked warning evidence",
            process=_safe_process(process),
            missing_warning_ids=sorted(missing_warning_ids),
        )
    return [warning for warning in payload if warning.get("id") in warning_ids]


def _load_verdicts() -> list[dict]:
    try:
        lines = VERDICT_LOG_PATH.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    try:
        records = [json.loads(line) for line in lines if line.strip()]
    except (TypeError, ValueError):
        _block("invalid verdict log", path=str(VERDICT_LOG_PATH))
    if not all(isinstance(record, dict) for record in records):
        _block("invalid verdict log", path=str(VERDICT_LOG_PATH))
    return records


def _latest_process_verdict(records: list[dict], process_id: str) -> dict | None:
    return next((record for record in reversed(records) if record.get("process_id") == process_id), None)


def _is_clean_no_work_closeout(process: dict | None) -> bool:
    """Whether a closeout completed only because the goal had no executable work."""
    outputs = process.get("outputs") if process else None
    return (
        process is not None
        and process.get("process_type") == "goal_closeout"
        and process.get("status") == "completed"
        and isinstance(outputs, dict)
        and outputs.get("no_executable_work") is True
        and outputs.get("full_closeout") is False
        and outputs.get("completion_authorized") is True
        and outputs.get("mode") == "completion"
        and outputs.get("gates") == {"closeout_completed": True}
        and outputs.get("warning_ids") in (None, [])
    )


def _skip_no_work_closeout(process_type: str) -> None:
    PACKET_PATH.unlink(missing_ok=True)
    print(f"SKIP: {process_type} completed (no executable work)")
    _append_event("phase_no_work_skip", process_type=process_type)


def _phase_action(process: dict | None, verdicts: list[dict]) -> str:
    """Determine action for a process: step, rerun, judge_existing, skip, or no_work_skip."""
    if process is None:
        return "step"
    status = process.get("status")
    judgeable_waiting = (
        status == "waiting_decision" and process.get("process_type") in JUDGEABLE_WAITING_TYPES
    )
    if status not in TERMINAL_STATUSES and not judgeable_waiting:
        return "step"

    fingerprint = _process_fingerprint(process)
    process_id = str(process.get("id"))

    if _is_clean_no_work_closeout(process):
        return "no_work_skip"

    if reusable_verdict(verdicts, process_id, fingerprint):
        return "skip"

    latest = _latest_process_verdict(verdicts, process_id)
    if latest is None:
        return "judge_existing"

    if latest.get("verdict") == "needs_human_judgment" and latest.get("fingerprint") == fingerprint:
        return "rerun"

    return "judge_existing"


def _process_fingerprint(process: dict) -> str:
    process_id = process.get("id")
    if (
        not process_id
        or process.get("process_type") not in PROCESS_ORDER
        or not isinstance(process.get("status"), str)
        or not isinstance(process.get("outputs"), dict)
    ):
        _block("invalid process output", process=_safe_process(process))
    try:
        return canonical_fingerprint(process["outputs"])
    except (TypeError, ValueError):
        _block("invalid process output", process=_safe_process(process))


def _is_low_risk_clarification(decision: dict) -> bool:
    return (
        decision.get("status") == "pending"
        and decision.get("authority") == "human"
        and str(decision.get("decision_key", "")).startswith("goal_definition:adaptive:")
        and not any(
            decision.get(key)
            for key in ("consequences", "created_warning_id", "related_gate_id", "related_action_id")
        )
    )


def _user_input_requirement(decisions: list[dict], checkpoint: dict) -> dict:
    checkpoint_ids = {
        str(item.get("id"))
        for item in checkpoint.get("items", [])
        if isinstance(item, dict) and item.get("id")
    }
    items = []
    for decision in decisions:
        if str(decision.get("id")) not in checkpoint_ids:
            continue
        items.append(
            {
                key: _redact(decision.get(key))
                for key in ("id", "question", "context", "options", "recommendation")
            }
            | {"delegation_allowed": _is_low_risk_clarification(decision)}
        )
    return {
        "status": "required",
        "items": items,
        "deferred_count": checkpoint.get("deferred_count", 0),
        "protocol": (
            "Ask the user with the listed question, options, and context. Only an explicitly delegated "
            "item marked delegation_allowed may use --answer-delegated; otherwise pause."
        ),
    }


def _orchestration_evidence(
    project_id: str,
    goal_id: str,
    process: dict,
    linked_warnings: list[dict],
) -> dict:
    process_type = process.get("process_type")
    return {
        "target": {"project_id": project_id, "goal_id": goal_id},
        "process": _redact(_safe_process(process)),
        "inputs": _redact(process.get("inputs")),
        "outputs": _redact(process.get("outputs")),
        "linked_warnings": _redact(linked_warnings),
        "phase": PHASES.get(process_type),
    }


def _packet(
    project_id: str,
    goal_id: str,
    process: dict,
    linked_warnings: list[dict],
    verdicts: list[dict],
    *,
    created_at: str | None = None,
    terminal_slice: list[str] | None = None,
    request_trace: list[dict] | None = None,
    user_input: dict | None = None,
) -> dict:
    process_id = str(process.get("id") or "")
    fingerprint = _process_fingerprint(process)
    prior = _latest_process_verdict(verdicts, process_id)
    prior_valid = bool(
        prior
        and prior.get("fingerprint") == fingerprint
        and prior.get("verdict") in REUSABLE_VERDICTS
    )
    prior_summary = (
        {
            key: prior.get(key)
            for key in ("recorded_at", "process_id", "fingerprint", "verdict", "spr_id")
            if prior.get(key) is not None
        }
        if prior
        else None
    )
    evidence = _orchestration_evidence(project_id, goal_id, process, linked_warnings)
    packet = {
        "schema_version": 2,
        "created_at": created_at or _now(),
        **evidence,
        "fingerprint": fingerprint,
        "evidence_fingerprint": canonical_fingerprint(evidence),
        "prior_verdict": {"record": prior_summary, "valid": prior_valid} if prior else None,
        "acceptance_checks": list(evidence["phase"]["acceptance_criteria"]),
        "observations": {
            "terminal_slice": [_redact_text(line) for line in terminal_slice or []],
            "request_trace": _redact(request_trace or []),
        },
        "user_input": _redact(user_input),
        "decision": {
            "status": "user_input_required" if user_input else "judgment_required",
            "allowed_verdicts": [] if user_input else list(VERDICTS),
        },
        "next_action": (
            "Ask the user and pause; only use --answer-delegated after explicit delegation."
            if user_input
            else "Inspect this packet, then run --record pass|minor|needs_human_judgment "
            "--reason '<concise evidence-based reason>'."
        ),
    }
    packet["packet_fingerprint"] = canonical_fingerprint(packet)
    return packet


def _emit_user_input(packet: dict) -> int:
    global _INVOCATION_PACKET

    packet = _redact(packet)
    PACKET_PATH.parent.mkdir(parents=True, exist_ok=True)
    PACKET_PATH.write_text(json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _INVOCATION_PACKET = packet
    for item in packet["user_input"]["items"]:
        print(f"USER INPUT REQUIRED: {item['question']}")
        print(f"OPTIONS: {json.dumps(item.get('options') or ['free-text'], ensure_ascii=False)}")
        print(f"CONTEXT: {item.get('context') or 'No additional context.'}")
    print(f"PACKET: {PACKET_PATH}")
    print("PAUSED: do not use browser automation or guess an answer")
    return HUMAN_JUDGMENT_REQUIRED


def _delegated_answer_request(
    packet: dict,
    answer: str,
    reason: str = "Explicitly delegated baseline clarification",
) -> tuple[str, dict]:
    try:
        decision_id, selected_option = answer.split(":", 1)
    except ValueError:
        _block("--answer-delegated must be DECISION_ID:ANSWER")
    if not decision_id or not selected_option.strip():
        _block("--answer-delegated must be DECISION_ID:ANSWER")
    item = next(
        (
            item
            for item in (packet.get("user_input") or {}).get("items", [])
            if str(item.get("id")) == decision_id
        ),
        None,
    )
    if not item or item.get("delegation_allowed") is not True:
        _block("answer is not an explicitly delegated low-risk clarification from the current packet")
    options = item.get("options") or []
    if options and selected_option not in options:
        _block("delegated answer is not one of the packet options", options=options)
    target = packet.get("target") or {}
    project_id, goal_id = target.get("project_id"), target.get("goal_id")
    if not project_id or not goal_id:
        _block("invalid judgment packet", path=str(PACKET_PATH))
    return (
        f"/api/v1/projects/{project_id}/orchestration/goals/{goal_id}/decisions/{decision_id}/answer",
        {"selected_option": selected_option, "reason": reason},
    )


def _emit_packet(packet: dict) -> int:
    global _INVOCATION_PACKET

    packet = _redact(packet)
    PACKET_PATH.parent.mkdir(parents=True, exist_ok=True)
    PACKET_PATH.write_text(
        json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _INVOCATION_PACKET = packet
    process = packet["process"]
    print(
        f"JUDGMENT REQUIRED: {process['process_type']} status={process.get('status')} "
        f"process_id={process['id']}"
    )
    print(f"PACKET: {PACKET_PATH}")
    print("NEXT: inspect the packet and record pass, minor, or needs_human_judgment with a reason")
    return JUDGMENT_REQUIRED


def _check_blockers(processes: list[dict]) -> dict | None:
    for process in current_processes(processes).values():
        if process.get("process_type") not in PROCESS_ORDER:
            continue
        reason = blocker_reason(process)
        if reason is None:
            continue
        if reason == "waiting for approval":
            if process.get("process_type") in JUDGEABLE_WAITING_TYPES:
                continue
            return process
        _block(reason, process=_safe_process(process))
    return None


def _user_input_packet(
    client: httpx.Client,
    project_id: str,
    goal_id: str,
    process: dict,
    verdicts: list[dict],
    terminal_start: int,
    trace_start: int,
) -> dict | None:
    base = f"/api/v1/projects/{project_id}/orchestration/goals/{goal_id}"
    decisions = _request(client, "GET", f"{base}/decisions", params={"status": "pending"})
    checkpoint = _request(client, "GET", f"{base}/decisions/checkpoint")
    if not isinstance(decisions, list) or not all(isinstance(item, dict) for item in decisions):
        _block("invalid pending decision list")
    if not isinstance(checkpoint, dict):
        _block("invalid decision checkpoint")
    user_input = _user_input_requirement(decisions, checkpoint)
    if not user_input["items"]:
        return None
    return _packet(
        project_id,
        goal_id,
        process,
        _warnings(client, project_id, goal_id, process),
        verdicts,
        terminal_slice=_TERMINAL_LINES[terminal_start:],
        request_trace=_REQUEST_TRACE[trace_start:],
        user_input=user_input,
    )


def _advance(
    client: httpx.Client,
    project_id: str,
    goal_id: str,
    process_types: tuple[str, ...],
    terminal_start: int,
    trace_start: int,
) -> int:
    """Advance baseline processes, iterating through given process_types."""
    processes = _processes(client, project_id, goal_id)
    waiting = _check_blockers(processes)
    current = current_processes(processes)
    verdicts = _load_verdicts()

    # Handle user input requirement from blockers
    if waiting:
        packet = _user_input_packet(
            client, project_id, goal_id, waiting, verdicts, terminal_start, trace_start
        )
        if packet:
            _write_checkpoint(project_id, goal_id, waiting)
            _append_event("user_input_required", process=_safe_process(waiting))
            return _emit_user_input(packet)

    step_path = f"/api/v1/projects/{project_id}/orchestration/goals/{goal_id}/debug/baseline/step"
    rerun_path = f"/api/v1/projects/{project_id}/orchestration/goals/{goal_id}/debug/baseline/rerun-last"

    for process_type in process_types:
        process = current.get(process_type)
        action = _phase_action(process, verdicts)

        if action == "no_work_skip":
            _skip_no_work_closeout(process_type)
            continue

        if action == "skip":
            if process:
                fingerprint = _process_fingerprint(process)
                reused = reusable_verdict(verdicts, str(process.get("id")), fingerprint)
                print(
                    f"VERDICT REUSED: {process_type} verdict={reused['verdict']} "
                    f"process_id={process.get('id')} fingerprint={fingerprint}"
                )
            continue

        if action == "judge_existing":
            # Terminal status with no reusable verdict
            packet = _packet(
                project_id,
                goal_id,
                process,
                _warnings(client, project_id, goal_id, process),
                verdicts,
                terminal_slice=_TERMINAL_LINES[terminal_start:],
                request_trace=_REQUEST_TRACE[trace_start:],
            )
            _write_checkpoint(project_id, goal_id, process)
            fingerprint = _process_fingerprint(process)
            _append_event("judgment_required", process=_safe_process(process), fingerprint=fingerprint)
            return _emit_packet(packet)

        # "step" or "rerun"
        path = step_path if action == "step" else rerun_path
        stepped = _request(
            client,
            "POST",
            path,
            json={"process_type": process_type},
            ignore_response=_closeout_not_ready if process_type == "goal_closeout" else None,
        )
        if stepped is None:
            print(f"SKIP: {process_type} not ready (preconditions unmet)")
            _append_event("phase_not_ready_skip", process_type=process_type)
            continue
        if not isinstance(stepped, dict):
            _block("invalid debug output", process_type=process_type)

        if (
            process_type == "effectiveness_review"
            and isinstance(stepped.get("process"), dict)
            and stepped["process"].get("status") == "idle"
        ):
            print(f"SKIP: {process_type} idle (no trigger conditions met)")
            _append_event("phase_idle_skip", process_type=process_type)
            continue

        refreshed = _processes(client, project_id, goal_id)
        waiting = _check_blockers(refreshed)
        refreshed_current = current_processes(refreshed)
        process = refreshed_current.get(process_type)
        if process is None:
            _block("invalid debug output", process_type=process_type)

        _process_fingerprint(process)
        if _is_clean_no_work_closeout(process):
            _skip_no_work_closeout(process_type)
            continue

        if waiting:
            packet = _user_input_packet(
                client, project_id, goal_id, waiting, verdicts, terminal_start, trace_start
            )
            if packet:
                _write_checkpoint(project_id, goal_id, waiting)
                _append_event("user_input_required", process=_safe_process(waiting))
                return _emit_user_input(packet)

        packet = _packet(
            project_id,
            goal_id,
            process,
            _warnings(client, project_id, goal_id, process),
            verdicts,
            terminal_slice=_TERMINAL_LINES[terminal_start:],
            request_trace=_REQUEST_TRACE[trace_start:],
        )
        _write_checkpoint(project_id, goal_id, process)

        event_kind = "step_judgment_required" if action == "step" else "rerun_judgment_required"
        _append_event(
            event_kind,
            process=_safe_process(process),
            fingerprint=packet["fingerprint"],
        )
        return _emit_packet(packet)

    # Completion path
    _write_checkpoint(project_id, goal_id)
    _append_event("complete", project_id=project_id, goal_id=goal_id, process_types=list(process_types))
    print(f"COMPLETE: reusable judgment for {', '.join(process_types)}")
    return 0


def _step(
    client: httpx.Client,
    project_id: str,
    goal_id: str,
    terminal_start: int,
    trace_start: int,
) -> int:
    return _advance(client, project_id, goal_id, PROCESS_ORDER, terminal_start, trace_start)


def _read_packet() -> dict:
    try:
        packet = json.loads(PACKET_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _block("judgment packet not found", path=str(PACKET_PATH))
    except (TypeError, ValueError):
        _block("invalid judgment packet", path=str(PACKET_PATH))
    if not isinstance(packet, dict):
        _block("invalid judgment packet", path=str(PACKET_PATH))
    return packet


def _create_minor_spr(packet: dict, reason: str) -> int:
    process = packet["process"]
    evidence = {"packet": packet, "reason": reason, "verdict_log": str(VERDICT_LOG_PATH)}
    with sqlite3.connect(Path(__file__).resolve().parents[2] / "bugs.db") as database:
        cursor = database.execute(
            "INSERT INTO bugs (title, status, priority, description, analysis) "
            "VALUES (?, 'Open', 'Medium', ?, ?)",
            (
                f"Baseline minor judgment: {process.get('process_type')}",
                json.dumps(evidence, ensure_ascii=False, sort_keys=True),
                "Agent judged the current HuddleRoom output as usable with a minor issue; no fix applied.",
            ),
        )
        return int(cursor.lastrowid)


def _record(
    client: httpx.Client,
    project_id: str,
    goal_id: str,
    verdict: str,
    reason: str,
) -> int:
    packet = _read_packet()
    if packet.get("target") != {"project_id": project_id, "goal_id": goal_id}:
        _block("packet target no longer matches current HuddleRoom target", path=str(PACKET_PATH))
    packet_process = packet.get("process")
    if not isinstance(packet_process, dict):
        _block("invalid judgment packet", path=str(PACKET_PATH))
    process_type = packet_process.get("process_type")
    if process_type not in PROCESS_ORDER:
        _block("invalid judgment packet", path=str(PACKET_PATH))
    processes = _processes(client, project_id, goal_id)
    if waiting := _check_blockers(processes):
        _block("waiting for approval", process=_safe_process(waiting))
    current = current_processes(processes).get(process_type)
    process_id = str(packet_process.get("id") or "")
    fingerprint = packet.get("fingerprint")
    try:
        output_fingerprint = canonical_fingerprint(packet.get("outputs"))
        current_fingerprint = _process_fingerprint(current) if current else None
    except (TypeError, ValueError):
        _block("invalid judgment packet", path=str(PACKET_PATH))
    if not process_id or not isinstance(fingerprint, str) or output_fingerprint != fingerprint:
        _block("invalid judgment packet", path=str(PACKET_PATH))
    if current is None or str(current.get("id")) != process_id or current_fingerprint != fingerprint:
        _block("packet no longer matches current HuddleRoom process", process=_safe_process(current))

    recorded_packet_fingerprint = packet.get("packet_fingerprint")
    unsigned_packet = {key: value for key, value in packet.items() if key != "packet_fingerprint"}
    try:
        packet_is_valid = (
            isinstance(recorded_packet_fingerprint, str)
            and canonical_fingerprint(unsigned_packet) == recorded_packet_fingerprint
        )
    except (TypeError, ValueError):
        _block("invalid judgment packet", path=str(PACKET_PATH))
    if not packet_is_valid:
        _block("invalid judgment packet", path=str(PACKET_PATH))
    packet_evidence = {
        key: packet.get(key)
        for key in ("target", "process", "inputs", "outputs", "linked_warnings", "phase")
    }
    try:
        packet_evidence_fingerprint = canonical_fingerprint(packet_evidence)
        current_evidence_fingerprint = canonical_fingerprint(
            _orchestration_evidence(project_id, goal_id, current, _warnings(client, project_id, goal_id, current))
        )
    except (TypeError, ValueError):
        _block("invalid current HuddleRoom evidence", process=_safe_process(current))
    if (
        packet.get("evidence_fingerprint") != packet_evidence_fingerprint
        or current_evidence_fingerprint != packet_evidence_fingerprint
    ):
        _block("packet no longer matches current HuddleRoom evidence", process=_safe_process(current))

    record = {
        "recorded_at": _now(),
        "project_id": project_id,
        "goal_id": goal_id,
        "process_type": process_type,
        "process_id": process_id,
        "fingerprint": fingerprint,
        "verdict": verdict,
        "reason": reason,
    }
    spr_id: int | None = None
    if verdict == "minor":
        try:
            spr_id = _create_minor_spr(packet, reason)
        except sqlite3.Error as exc:
            _block("could not create minor SPR", error=type(exc).__name__)
        record["spr_id"] = spr_id
    _append_jsonl(VERDICT_LOG_PATH, record)
    _append_event(
        "verdict_recorded",
        project_id=project_id,
        goal_id=goal_id,
        process_type=process_type,
        process_id=process_id,
        fingerprint=fingerprint,
        verdict=verdict,
        spr_id=spr_id,
    )
    if verdict == "needs_human_judgment":
        print(f"HUMAN JUDGMENT REQUIRED: {process_type} process_id={process_id}")
        print(f"VERDICT LOG: {VERDICT_LOG_PATH}")
        return HUMAN_JUDGMENT_REQUIRED
    suffix = f" spr_id={spr_id}" if spr_id else ""
    print(f"VERDICT RECORDED: {process_type} verdict={verdict} process_id={process_id}{suffix}")
    print(f"VERDICT LOG: {VERDICT_LOG_PATH}")
    return 0


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--step", action="store_true", help="advance all baseline phases (alias of --run), stepping or rerunning as needed, and emit a packet")
    action.add_argument("--run", action="store_true", help="advance every not-yet-passed baseline phase, stepping or rerunning as needed, and emit a packet for the first one that needs judgment")
    action.add_argument("--phase", choices=PROCESS_ORDER, help="do the same as --run but scoped to one phase")
    action.add_argument("--reset", action="store_true", help="clear all runner state and artifacts, then exit")
    action.add_argument("--record", choices=VERDICTS, help="record judgment for the current packet")
    action.add_argument(
        "--answer-delegated",
        metavar="DECISION_ID:ANSWER",
        help="answer a low-risk clarification from the current packet after explicit user delegation",
    )
    parser.add_argument("--reason", help="concise reason for --record or --answer-delegated")
    parser.add_argument(
        "--controlled-server",
        action="store_true",
        help="start an isolated HuddleRoom CLI child for this invocation and always stop it",
    )
    args = parser.parse_args(argv)
    if (args.record or args.answer_delegated) and not (args.reason and args.reason.strip()):
        parser.error("--record and --answer-delegated require a non-empty --reason")
    if args.reason and not (args.record or args.answer_delegated):
        parser.error("--reason is only valid with --record or --answer-delegated")
    if args.reset and args.controlled_server:
        parser.error("--reset does not use a server")

    if args.reset:
        for path in (CHECKPOINT_PATH, VERDICT_LOG_PATH, PACKET_PATH, EVENT_LOG_PATH, HTTP_TRACE_PATH, ARTIFACT_DIR / "server.log"):
            path.unlink(missing_ok=True)
        shutil.rmtree(ARTIFACT_DIR / "invocations", ignore_errors=True)
        print(f"RESET: cleared {ARTIFACT_DIR}")
        return 0

    base_url = os.environ.get("HUDDLEROOM_URL") or os.environ.get("RALLY_URL")
    _ACTIVE_TARGET.clear()
    _REQUEST_TRACE.clear()
    _TERMINAL_LINES.clear()
    global _INVOCATION_PACKET
    _INVOCATION_PACKET = None
    invocation_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex}"
    states: dict[str, dict] = {}
    try:
        if not base_url and not args.controlled_server:
            _block("HUDDLEROOM_URL is required")
        server = _controlled_server() if args.controlled_server else nullcontext(base_url.rstrip("/"))
        with server as active_url:
            terminal_start = len(_TERMINAL_LINES)
            trace_start = len(_REQUEST_TRACE)
            with httpx.Client(base_url=active_url, timeout=300) as client:
                project_id, goal_id = _resolve_target(client)
                _ACTIVE_TARGET.update(project_id=project_id, goal_id=goal_id)
                states["before"] = _capture_state(client, project_id, goal_id)
                try:
                    if args.step or args.run:
                        return _step(client, project_id, goal_id, terminal_start, trace_start)
                    if args.phase:
                        return _advance(client, project_id, goal_id, (args.phase,), terminal_start, trace_start)
                    if args.answer_delegated:
                        packet = _read_packet()
                        recorded = packet.get("packet_fingerprint")
                        unsigned = {key: value for key, value in packet.items() if key != "packet_fingerprint"}
                        if not isinstance(recorded, str) or canonical_fingerprint(unsigned) != recorded:
                            _block("invalid judgment packet", path=str(PACKET_PATH))
                        if packet.get("target") != {"project_id": project_id, "goal_id": goal_id}:
                            _block("packet target no longer matches current HuddleRoom target", path=str(PACKET_PATH))
                        path, payload = _delegated_answer_request(
                            packet, args.answer_delegated, args.reason.strip()
                        )
                        _request(client, "POST", path, json=payload)
                        _append_event("delegated_answer_recorded", decision_id=args.answer_delegated.split(":", 1)[0])
                        print(f"DELEGATED ANSWER RECORDED: {args.answer_delegated.split(':', 1)[0]}")
                        return 0
                    return _record(client, project_id, goal_id, args.record, args.reason.strip())
                finally:
                    states["after"] = _capture_state(client, project_id, goal_id)
    except Blocked as exc:
        _write_diagnostic_packet(str(exc))
        print(f"BLOCKED: {exc}", file=sys.stderr)
        print(f"EVENT LOG: {EVENT_LOG_PATH}", file=sys.stderr)
        return 1
    finally:
        if args.controlled_server:
            _write_server_log()
        _persist_invocation_evidence(invocation_id, states)


if __name__ == "__main__":
    raise SystemExit(run())
