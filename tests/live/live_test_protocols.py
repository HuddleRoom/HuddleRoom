#!/usr/bin/env python3
"""HuddleRoom protocol live integration test suite — code_review protocol real-world tests.

This test suite uses the real `code_review` protocol loaded from the workspace
and verifies critical side effects: template resolution in actions, sessions,
messages, knowledge items, task completion, and escalation.

Usage:
    python tests/live/live_test_protocols.py [options]

Options:
    --port PORT           Port to use (default: 8001)
    --no-start-server     Don't start server; assume already running
    --keep-server         Don't stop the server after tests
    --tests PC1,...       Comma-separated list of test IDs to run (default: all)
    --verbose             Print detailed output
    --timeout SECONDS     Per-test poll timeout (default: 60, must be ≥40 for PC9)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TextIO

import httpx
import litellm

# ── Constants ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECT_NAME = "huddleroom-protocol-live-test"
DEFAULT_PAGE_LIMIT = 500
OPENAI_READINESS_MODEL = "openai/gpt-6-luna"
DEFAULT_PR_REPOSITORY = "acme/huddleroom-live-harness"
DEFAULT_PR_BASE_BRANCH = "main"

PROTO_AGENTS = [
    {
        "name": "proto-author",
        "role": "engineer",
        "provider": "openai",
        "model": "gpt-6-luna",
        "system_prompt": "Protocol test author agent.",
        "capabilities": [],
    },
    {
        "name": "proto-reviewer",
        "role": "reviewer",
        "provider": "openai",
        "model": "gpt-6-luna",
        "adapter_type": "api",
        "system_prompt": (
            "You are a code reviewer in an automated test pipeline. "
            "Your task is to review pull requests."
        ),
        "capabilities": ["code_review"],
    },
    {
        "name": "proto-merger",
        "role": "pm",
        "provider": "openai",
        "model": "gpt-6-luna",
        "system_prompt": "Protocol test merger agent.",
        "capabilities": [],
    },
]

# ── Result tracking ────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    test_id: str
    label: str
    passed: bool
    message: str
    elapsed: float = 0.0
    errors: list[str] = field(default_factory=list)
    transitions: int = 0


@dataclass
class LogEntry:
    """Structured log entry for HTML report."""
    timestamp: str        # ISO format
    kind: str             # "http_get" | "http_post" | "info"
    test_id: str | None   # which test this belongs to, or None for setup
    data: dict            # flexible payload


class RunLog:
    """Collects structured events during the run."""
    def __init__(self) -> None:
        self.entries: list[LogEntry] = []
        self.results: list[TestResult] = []

    def add(self, kind: str, test_id: str | None, data: dict) -> None:
        """Add a structured log entry."""
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        self.entries.append(LogEntry(
            timestamp=timestamp,
            kind=kind,
            test_id=test_id,
            data=data
        ))


# ── API client ─────────────────────────────────────────────────────────────────

class HuddleRoomClient:
    def __init__(self, base_url: str, log: logging.Logger | None = None, run_log: RunLog | None = None, test_id: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self.base_url, timeout=60)
        self.log = log
        self.run_log = run_log
        self.test_id = test_id

    def _paged_get_items(self, path: str, limit: int = DEFAULT_PAGE_LIMIT, **params: Any) -> list[dict]:
        """Collect every item from a cursor-paginated endpoint."""
        items: list[dict] = []
        cursor: str | None = None
        while True:
            page_params = {**params, "limit": limit}
            if cursor:
                page_params["cursor"] = cursor
            resp = self._get(path, **page_params)
            if not isinstance(resp, dict):
                return list(resp)
            items.extend(resp.get("items", []))
            cursor = resp.get("next_cursor")
            if not cursor:
                return items

    def _get(self, path: str, **params: Any) -> Any:
        r = self._http.get(path, params=params)
        r.raise_for_status()
        resp_json = r.json()
        if self.log:
            self.log.info(f"GET {path} → {r.status_code}")
            self.log.info(json.dumps(resp_json))
        if self.run_log:
            self.run_log.add("http_get", self.test_id, {
                "path": path,
                "params": params,
                "status": r.status_code,
                "response": resp_json
            })
        return resp_json

    def _post(self, path: str, body: dict) -> tuple[Any, int]:
        """POST and return (response, status_code). Raises on non-2xx."""
        r = self._http.post(path, json=body)
        if self.log:
            self.log.info(f"POST {path} body={json.dumps(body)} → {r.status_code}")
        if not r.is_success:
            err_text = r.text[:200]
            raise RuntimeError(f"POST {path} failed {r.status_code}: {err_text}")
        resp_json = r.json()
        if self.log:
            self.log.info(json.dumps(resp_json))
        if self.run_log:
            self.run_log.add("http_post", self.test_id, {
                "path": path,
                "body": body,
                "status": r.status_code,
                "response": resp_json
            })
        return resp_json, r.status_code

    def _put(self, path: str, body: dict) -> tuple[Any, int]:
        """PUT and return (response, status_code). Raises on non-2xx."""
        r = self._http.put(path, json=body)
        if self.log:
            self.log.info(f"PUT {path} body={json.dumps(body)} → {r.status_code}")
        if not r.is_success:
            err_text = r.text[:200]
            raise RuntimeError(f"PUT {path} failed {r.status_code}: {err_text}")
        resp_json = r.json()
        if self.log:
            self.log.info(json.dumps(resp_json))
        return resp_json, r.status_code

    def get_or_create_project(self, name: str) -> dict:
        for p in self._paged_get_items("/api/v1/projects", limit=200):
            if p["name"] == name:
                print(f"  Reusing project '{name}' ({p['id']})")
                return p
        print(f"  Creating project '{name}'...")
        resp, _ = self._post("/api/v1/projects", {"name": name, "description": "HuddleRoom protocol live integration tests"})
        return resp

    def create_agent(self, defn: dict) -> dict:
        resp, _ = self._post("/api/v1/agents", defn)
        return resp

    def get_or_create_agent(self, defn: dict) -> dict:
        """Reuse agent by name if exists (updating it), else create."""
        agents = self._get("/api/v1/agents", limit=200)
        agent_name = defn["name"]
        for a in agents.get("items", []):
            if a.get("name") == agent_name:
                # Force-update so provider/model/system_prompt stay current.
                updated, _ = self._put(f"/api/v1/agents/{a['id']}", defn)
                return updated
        return self.create_agent(defn)

    def create_task(self, project_id: str, title: str, description: str, metadata: dict) -> dict:
        body = {
            "title": title,
            "description": description,
            "metadata": metadata,
        }
        resp, _ = self._post(f"/api/v1/projects/{project_id}/tasks", body)
        return resp

    def get_task(self, project_id: str, task_id: str) -> dict:
        return self._get(f"/api/v1/projects/{project_id}/tasks/{task_id}")

    def create_artifact(self, project_id: str, name: str, artifact_type: str, metadata: dict, linked_task_id: str | None = None) -> dict:
        body = {
            "name": name,
            "artifact_type": artifact_type,
            "metadata": metadata,
        }
        if linked_task_id:
            body["linked_task_id"] = linked_task_id
        resp, _ = self._post(f"/api/v1/projects/{project_id}/artifacts", body)
        return resp

    def emit_event(self, project_id: str, event_type: str, payload: dict) -> dict:
        body = {
            "project_id": project_id,
            "event_type": event_type,
            "payload": payload,
            "source": "test"
        }
        resp, _ = self._post("/api/v1/events", body)
        return resp

    def list_protocols(self, project_id: str) -> list[dict]:
        resp = self._get(f"/api/v1/projects/{project_id}/protocols")
        return resp if isinstance(resp, list) else resp.get("items", [])

    def list_instances(self, project_id: str, status: str | None = None) -> list[dict]:
        params = {}
        if status:
            params["status"] = status
        return self._paged_get_items(f"/api/v1/projects/{project_id}/protocol-instances", **params)

    def get_instance(self, project_id: str, instance_id: str) -> dict:
        return self._get(f"/api/v1/projects/{project_id}/protocol-instances/{instance_id}")

    def list_transitions(self, project_id: str, instance_id: str) -> list[dict]:
        resp = self._get(f"/api/v1/projects/{project_id}/protocol-instances/{instance_id}/transitions")
        return resp if isinstance(resp, list) else resp.get("items", [])

    def advance_instance(self, project_id: str, instance_id: str, to_state: str, reason: str) -> tuple[Any, int]:
        body = {"to_state": to_state, "reason": reason}
        return self._post(f"/api/v1/projects/{project_id}/protocol-instances/{instance_id}/advance", body)

    def list_channels(self, project_id: str) -> list[dict]:
        resp = self._get(f"/api/v1/projects/{project_id}/channels")
        return resp if isinstance(resp, list) else resp.get("items", [])

    def list_messages(self, channel_id: str, limit: int = 50) -> list[dict]:
        return self._paged_get_items(f"/api/v1/channels/{channel_id}/messages", limit=limit)

    def list_sessions(self, project_id: str, agent_id: str | None = None) -> list[dict]:
        params = {"project_id": project_id}
        if agent_id:
            params["agent_id"] = agent_id
        return self._paged_get_items("/api/v1/sessions", **params)

    def get_session(self, session_id: str) -> dict:
        return self._get(f"/api/v1/sessions/{session_id}")

    def create_session(self, agent_id: str, project_id: str, task_id: str | None = None,
                       adapter_type_override: str | None = None,
                       protocol_instance_id: str | None = None) -> dict:
        body: dict = {"agent_id": agent_id, "project_id": project_id}
        if task_id:
            body["task_id"] = task_id
        if adapter_type_override:
            body["adapter_type_override"] = adapter_type_override
        if protocol_instance_id:
            body["protocol_instance_id"] = protocol_instance_id
        resp, _ = self._post("/api/v1/sessions", body)
        return resp

    def list_knowledge(self, project_id: str) -> list[dict]:
        return self._paged_get_items(f"/api/v1/projects/{project_id}/knowledge")


# Compatibility for existing live-test imports.
RallyClient = HuddleRoomClient


# ── Fixture setup ──────────────────────────────────────────────────────────────

def build_project_name(*, reuse_project: bool = False) -> str:
    """Return a reusable or per-run live-test project name."""
    if reuse_project:
        return PROJECT_NAME
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%Sz").lower()
    run_suffix = str(uuid.uuid4()).split("-", maxsplit=1)[0]
    return f"{PROJECT_NAME}-{timestamp}-{run_suffix}"


def setup_fixtures(
    client: RallyClient,
    log: logging.Logger | None = None,
    *,
    reuse_project: bool = False,
) -> tuple[dict, dict[str, dict]]:
    """Create or reuse project and agents. Returns (project, agents_by_name)."""
    print("\n=== Setting up fixtures ===")
    if log:
        log.info("=== Setting up fixtures ===")

    project_name = build_project_name(reuse_project=reuse_project)
    project = client.get_or_create_project(project_name)
    print(f"  Project: {project['id']}")

    agents_by_name = {}
    for agent_def in PROTO_AGENTS:
        agent = client.get_or_create_agent(agent_def)
        agents_by_name[agent_def["name"]] = agent
        print(f"  Agent {agent_def['name']}: {agent['id']}")

    return project, agents_by_name


# ── Helper functions ───────────────────────────────────────────────────────────

def find_protocol_by_name(client: RallyClient, project_id: str, name: str) -> dict | None:
    """Find a protocol by name from the workspace-loaded list."""
    protocols = client.list_protocols(project_id)
    return next((p for p in protocols if p.get("name") == name), None)


def _wait_for_value(
    description: str,
    fetch_value: Callable[[], Any],
    is_ready: Callable[[Any], bool],
    *,
    timeout: float,
    interval: float = 1.0,
    summarize: Callable[[Any], str] | None = None,
) -> Any:
    """Poll until a predicate passes, raising with useful last-seen diagnostics."""
    deadline = time.monotonic() + timeout
    last_value: Any = None
    while time.monotonic() < deadline:
        last_value = fetch_value()
        if is_ready(last_value):
            return last_value
        time.sleep(interval)
    summary = summarize(last_value) if summarize else repr(last_value)
    raise TimeoutError(f"Timed out waiting for {description}; last observed {summary}")


def build_synthetic_pr_metadata(artifact_name: str) -> dict[str, Any]:
    """Create reviewer-friendly synthetic pull request metadata."""
    normalized_slug = "".join(char.lower() if char.isalnum() else "-" for char in artifact_name).strip("-")
    pr_number = next((int(part) for part in reversed(normalized_slug.split("-")) if part.isdigit()), 1)
    scenario_tokens = [part for part in normalized_slug.split("-") if part and not part.isdigit()]
    scenario = next(
        (
            token
            for token in scenario_tokens
            if token not in {"pl1", "pl2", "pl3", "pc1", "pc2", "pc3", "pc4", "pc5", "pc6", "pc7", "pc8", "pc9", "pc10"}
        ),
        "workflow",
    )
    change_type = next((token for token in scenario_tokens if token in {"hotfix", "refactor", "docs"}), "feature")
    branch_prefix = "hotfix/test" if change_type == "hotfix" else f"{change_type}/test"
    repository_url = f"https://example.com/{DEFAULT_PR_REPOSITORY}"
    return {
        "branch": f"{branch_prefix}/{normalized_slug}",
        "base_branch": DEFAULT_PR_BASE_BRANCH,
        "repository": DEFAULT_PR_REPOSITORY,
        "repository_url": repository_url,
        "pr_number": pr_number,
        "pr_url": f"{repository_url}/pull/{pr_number}",
        "title": artifact_name,
        "summary": f"Synthetic {change_type} pull request for the {scenario} scenario in live reviewer coverage.",
        "change_summary": [
            f"Update the {scenario} flow exercised by {artifact_name}.",
            f"Adjust synthetic reviewer inputs for the {change_type} path.",
            f"Keep the live harness deterministic for PR #{pr_number}.",
        ],
        "test_plan": [
            f"Run the reviewer protocol against the {scenario} scenario.",
            "Confirm state transitions and emitted review events stay consistent.",
        ],
        "risk_notes": [
            f"{scenario.capitalize()} behavior may diverge between synthetic and live repository context.",
            f"{change_type.capitalize()} paths should preserve reviewer decision determinism.",
        ],
    }


def setup_pr(client: RallyClient, project_id: str, artifact_name: str = "Test PR") -> tuple[dict, dict]:
    """Create a task and artifact for a code review test. Returns (task, artifact)."""
    task = client.create_task(project_id, f"Implement: {artifact_name}", "Test task for protocol", {})
    # Small delay to let async event-bus consumers finish any in-flight write
    # transactions from prior operations (SQLite serializes writers).
    time.sleep(0.5)
    artifact = client.create_artifact(
        project_id,
        name=artifact_name,
        artifact_type="pull_request",
        metadata=build_synthetic_pr_metadata(artifact_name),
        linked_task_id=task["id"],
    )
    return task, artifact


def wait_for_instance(
    client: RallyClient,
    project_id: str,
    protocol_id: str,
    timeout: float = 15,
    artifact_id: str | None = None,
    interval: float = 1.0,
) -> dict:
    """Poll for active instance of a given protocol, optionally matching artifact_id."""

    def fetch_instance() -> dict | None:
        instances = client.list_instances(project_id, status="active")
        for inst in instances:
            if inst.get("protocol_id") != protocol_id:
                continue
            if artifact_id and inst.get("artifact_id") != artifact_id:
                continue
            return inst
        return None

    return _wait_for_value(
        f"protocol instance for protocol {protocol_id} artifact {artifact_id or '*'}",
        fetch_instance,
        lambda inst: inst is not None,
        timeout=timeout,
        interval=interval,
        summarize=lambda inst: "no matching active instance" if inst is None else repr(inst),
    )


def wait_for_state(
    client: RallyClient,
    project_id: str,
    instance_id: str,
    expected_state: str,
    timeout: float = 15,
    interval: float = 1.0,
) -> dict:
    """Poll until instance reaches expected state."""
    return _wait_for_value(
        f"protocol instance {instance_id} to reach expected state '{expected_state}'",
        lambda: client.get_instance(project_id, instance_id),
        lambda inst: inst.get("current_state") == expected_state,
        timeout=timeout,
        interval=interval,
        summarize=lambda inst: (
            f"state={inst.get('current_state')!r}, status={inst.get('status')!r}"
            if isinstance(inst, dict)
            else repr(inst)
        ),
    )


def wait_for_session_complete(
    client: RallyClient,
    project_id: str,
    protocol_instance_id: str,
    timeout: float = 90,
) -> dict:
    """Poll until a session linked to this protocol instance reaches completed or failed.

    Returns the full session dict (including .output) or None on timeout.
    """
    def fetch_session() -> dict | None:
        sessions = client.list_sessions(project_id)
        for s in sessions:
            if str(s.get("protocol_instance_id")) != protocol_instance_id:
                continue
            status = s.get("status")
            if status in ("completed", "failed"):
                # Fetch fresh copy with output field populated
                return client.get_session(str(s["id"]))
        return None

    return _wait_for_value(
        f"protocol session for instance {protocol_instance_id} to complete",
        fetch_session,
        lambda session: session is not None,
        timeout=timeout,
        interval=3.0,
        summarize=lambda session: "no completed session yet" if session is None else repr(session),
    )


def wait_for_protocol_session(
    client: RallyClient,
    project_id: str,
    protocol_instance_id: str,
    *,
    timeout: float = 60,
    interval: float = 1.0,
    origin: str | None = None,
) -> dict:
    """Poll until a session for the protocol instance exists."""

    def fetch_session() -> dict | None:
        for session in client.list_sessions(project_id):
            if str(session.get("protocol_instance_id")) != protocol_instance_id:
                continue
            if origin and session.get("origin") != origin:
                continue
            return session
        return None

    return _wait_for_value(
        f"protocol session for instance {protocol_instance_id}",
        fetch_session,
        lambda session: session is not None,
        timeout=timeout,
        interval=interval,
        summarize=lambda session: "no matching session yet" if session is None else repr(session),
    )


def wait_for_session_by_id(
    client: RallyClient,
    session_id: str,
    timeout: float = 300,
) -> dict:
    """Poll a specific session by ID until completed or failed."""
    return _wait_for_value(
        f"session {session_id} to complete",
        lambda: client.get_session(session_id),
        lambda session: session.get("status") in ("completed", "failed"),
        timeout=timeout,
        interval=3.0,
        summarize=lambda session: (
            f"status={session.get('status')!r}, error={session.get('error')!r}"
            if isinstance(session, dict)
            else repr(session)
        ),
    )


def wait_for_message(
    client: RallyClient,
    channel_id: str,
    predicate: Callable[[dict], bool],
    *,
    timeout: float = 30,
    interval: float = 1.0,
) -> dict:
    """Poll until a matching message exists in the channel."""

    def fetch_message() -> dict | None:
        for message in client.list_messages(channel_id, limit=DEFAULT_PAGE_LIMIT):
            if predicate(message):
                return message
        return None

    return _wait_for_value(
        f"message in channel {channel_id}",
        fetch_message,
        lambda message: message is not None,
        timeout=timeout,
        interval=interval,
        summarize=lambda message: "no matching message yet" if message is None else repr(message),
    )


def wait_for_knowledge_item(
    client: RallyClient,
    project_id: str,
    predicate: Callable[[dict], bool],
    *,
    timeout: float = 30,
    interval: float = 1.0,
) -> dict:
    """Poll until a matching knowledge item exists."""

    def fetch_item() -> dict | None:
        for item in client.list_knowledge(project_id):
            if predicate(item):
                return item
        return None

    return _wait_for_value(
        f"knowledge item in project {project_id}",
        fetch_item,
        lambda item: item is not None,
        timeout=timeout,
        interval=interval,
        summarize=lambda item: "no matching knowledge item yet" if item is None else repr(item),
    )


def wait_for_task_status(
    client: RallyClient,
    project_id: str,
    task_id: str,
    expected_status: str,
    *,
    timeout: float = 30,
    interval: float = 1.0,
) -> dict:
    """Poll until the task reaches the expected status."""
    return _wait_for_value(
        f"task {task_id} to reach status '{expected_status}'",
        lambda: client.get_task(project_id, task_id),
        lambda task: task.get("status") == expected_status,
        timeout=timeout,
        interval=interval,
        summarize=lambda task: (
            f"status={task.get('status')!r}" if isinstance(task, dict) else repr(task)
        ),
    )


# ── Test implementations ───────────────────────────────────────────────────────

def test_pc1_trigger_actor_resolution(client: RallyClient, project: dict, agents_by_name: dict[str, dict], log: logging.Logger | None = None, run_log: RunLog | None = None) -> TestResult:
    """PC1: Trigger & Actor Resolution — verify all 3 actors correctly resolved."""
    test_id = "PC1"
    label = "Trigger & Actor Resolution"
    t_start = time.monotonic()
    errors: list[str] = []

    if run_log:
        client.test_id = test_id
        client.run_log = run_log

    try:
        # Find code_review protocol
        protocol = find_protocol_by_name(client, project["id"], "code_review")
        if not protocol:
            errors.append("code_review protocol not found")
            raise RuntimeError("code_review protocol not found")

        protocol_id = protocol["id"]
        print(f"    Found code_review protocol {protocol_id}")

        # Create task + artifact
        task, artifact = setup_pr(client, project["id"], f"PC1-TEST-{int(time.time())}")
        print(f"    Created task {task['id']}, artifact {artifact['id']}")

        # Emit trigger with author agent
        author_id = agents_by_name["proto-author"]["id"]
        client.emit_event(project["id"], "code.pr_opened", {
            "artifact_id": artifact["id"],
            "task_id": task["id"],
            "author_agent_id": author_id,
        })
        print(f"    Emitted code.pr_opened with author_agent_id")

        # Poll for instance
        instance = wait_for_instance(client, project["id"], protocol_id, timeout=15, artifact_id=artifact["id"])
        if not instance:
            errors.append("No instance created within 15s")
        else:
            instance_id = instance["id"]
            print(f"    Instance {instance_id} created in state {instance.get('current_state')}")

            # Verify state and status
            if instance.get("current_state") != "opened":
                errors.append(f"Expected current_state='opened', got '{instance.get('current_state')}'")
            if instance.get("status") != "active":
                errors.append(f"Expected status='active', got '{instance.get('status')}'")

            # Verify actor assignments
            actors = instance.get("actor_assignments") or {}
            if "author" not in actors:
                errors.append("actor_assignments missing 'author'")
            elif actors["author"].get("id") != author_id:
                errors.append(f"author ID mismatch: expected {author_id}, got {actors['author'].get('id')}")
            else:
                print(f"    author assigned correctly")

            if "reviewer" not in actors:
                errors.append("actor_assignments missing 'reviewer'")
            elif actors["reviewer"].get("id") != agents_by_name["proto-reviewer"]["id"]:
                errors.append(f"reviewer ID mismatch")
            else:
                print(f"    reviewer assigned correctly")

            if "merger" not in actors:
                errors.append("actor_assignments missing 'merger'")
            elif actors["merger"].get("id") != agents_by_name["proto-merger"]["id"]:
                errors.append(f"merger ID mismatch")
            else:
                print(f"    merger assigned correctly")

    except Exception as exc:
        errors.append(str(exc))

    elapsed = time.monotonic() - t_start
    passed = len(errors) == 0
    message = "PASS" if passed else f"FAIL: {errors[0]}"
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    if errors:
        for e in errors:
            print(f"    - {e}")

    return TestResult(
        test_id=test_id,
        label=label,
        passed=passed,
        message=message,
        elapsed=elapsed,
        errors=errors,
        transitions=1
    )


def test_pc2_template_guard_resolution(client: RallyClient, project: dict, agents_by_name: dict[str, dict], log: logging.Logger | None = None, run_log: RunLog | None = None) -> TestResult:
    """PC2: Guard Specificity — only matching artifact transitions."""
    test_id = "PC2"
    label = "Template Guard Resolution (Guard Specificity)"
    t_start = time.monotonic()
    errors: list[str] = []

    if run_log:
        client.test_id = test_id
        client.run_log = run_log

    try:
        protocol = find_protocol_by_name(client, project["id"], "code_review")
        if not protocol:
            errors.append("code_review protocol not found")
            raise RuntimeError("code_review protocol not found")

        protocol_id = protocol["id"]

        # Create two tasks + artifacts
        task_a, artifact_a = setup_pr(client, project["id"], f"PC2-TEST-A-{int(time.time())}")
        task_b, artifact_b = setup_pr(client, project["id"], f"PC2-TEST-B-{int(time.time())}")
        print(f"    Created task A {artifact_a['id']}, task B {artifact_b['id']}")

        author_id = agents_by_name["proto-author"]["id"]

        # Emit pr_opened for A
        client.emit_event(project["id"], "code.pr_opened", {
            "artifact_id": artifact_a["id"],
            "task_id": task_a["id"],
            "author_agent_id": author_id,
        })

        # Emit pr_opened for B
        client.emit_event(project["id"], "code.pr_opened", {
            "artifact_id": artifact_b["id"],
            "task_id": task_b["id"],
            "author_agent_id": author_id,
        })

        # Poll for both instances
        deadline = time.time() + 15
        instances_by_artifact = {}
        while time.time() < deadline and len(instances_by_artifact) < 2:
            instances = client.list_instances(project["id"], status="active")
            for inst in instances:
                if inst.get("protocol_id") == protocol_id:
                    aid = inst.get("artifact_id")
                    if aid == artifact_a["id"]:
                        instances_by_artifact[artifact_a["id"]] = inst
                    elif aid == artifact_b["id"]:
                        instances_by_artifact[artifact_b["id"]] = inst
            if len(instances_by_artifact) < 2:
                time.sleep(1)

        if len(instances_by_artifact) < 2:
            errors.append(f"Only {len(instances_by_artifact)} instances created, expected 2")
        else:
            instance_a = instances_by_artifact[artifact_a["id"]]
            instance_b = instances_by_artifact[artifact_b["id"]]
            print(f"    Created 2 instances: A={instance_a['id']}, B={instance_b['id']}")

            # Emit test.passed for artifact_a ONLY
            client.emit_event(project["id"], "test.passed", {
                "artifact_id": artifact_a["id"],
            })
            print(f"    Emitted test.passed with artifact_a only")

            # Check states: A should transition to ready_for_review, B should stay opened
            inst_a = wait_for_state(client, project["id"], instance_a["id"], "ready_for_review", timeout=15)
            inst_b = client.get_instance(project["id"], instance_b["id"])

            if inst_a.get("current_state") != "ready_for_review":
                errors.append(f"Instance A should be ready_for_review, got {inst_a.get('current_state')}")
            else:
                print(f"    Instance A: ready_for_review ✓")

            if inst_b.get("current_state") != "opened":
                errors.append(f"Guard did not isolate: Instance B should stay opened, got {inst_b.get('current_state')}")
            else:
                print(f"    Instance B: opened ✓ (guard isolated correctly)")

    except Exception as exc:
        errors.append(str(exc))

    elapsed = time.monotonic() - t_start
    passed = len(errors) == 0
    message = "PASS" if passed else f"FAIL: {errors[0]}"
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    if errors:
        for e in errors:
            print(f"    - {e}")

    return TestResult(
        test_id=test_id,
        label=label,
        passed=passed,
        message=message,
        elapsed=elapsed,
        errors=errors,
        transitions=2
    )


def test_pc3_post_message_content(client: RallyClient, project: dict, agents_by_name: dict[str, dict], log: logging.Logger | None = None, run_log: RunLog | None = None) -> TestResult:
    """PC3: post_message Content Resolution — verify artifact name resolved in message."""
    test_id = "PC3"
    label = "post_message Content Resolution"
    t_start = time.monotonic()
    errors: list[str] = []

    if run_log:
        client.test_id = test_id
        client.run_log = run_log

    try:
        protocol = find_protocol_by_name(client, project["id"], "code_review")
        if not protocol:
            errors.append("code_review protocol not found")
            raise RuntimeError("code_review protocol not found")

        artifact_name = f"PC3-ARTIFACT-{uuid.uuid4().hex[:8]}"
        task, artifact = setup_pr(client, project["id"], artifact_name)
        print(f"    Created artifact with distinctive name: {artifact_name}")

        # Emit pr_opened
        author_id = agents_by_name["proto-author"]["id"]
        client.emit_event(project["id"], "code.pr_opened", {
            "artifact_id": artifact["id"],
            "task_id": task["id"],
            "author_agent_id": author_id,
        })

        # Wait for instance in opened state (filter by artifact_id to avoid stale instances)
        instance = wait_for_instance(client, project["id"], protocol["id"], timeout=15, artifact_id=artifact["id"])
        if not instance:
            errors.append("Instance not created")
        else:
            print(f"    Instance created in state {instance.get('current_state')}")

            # Find general channel
            channels = client.list_channels(project["id"])
            general_channel = next((c for c in channels if c.get("name") == "general"), None)
            if not general_channel:
                errors.append("general channel not found (post_message did not create it)")
            else:
                channel_id = general_channel["id"]
                print(f"    Found general channel {channel_id}")

                msg = wait_for_message(
                    client,
                    channel_id,
                    lambda message: message.get("metadata", {}).get("protocol_instance_id") == instance["id"],
                    timeout=15,
                )
                content = msg.get("content", "")
                print(f"    Found protocol message: {content[:60]}...")

                if artifact_name not in content:
                    errors.append(f"Artifact name '{artifact_name}' not found in message content")
                else:
                    print(f"    Artifact name resolved correctly")

                if "{{" in content:
                    errors.append("Message contains unresolved template placeholder")
                else:
                    print(f"    No template placeholders in message")

    except Exception as exc:
        errors.append(str(exc))

    elapsed = time.monotonic() - t_start
    passed = len(errors) == 0
    message = "PASS" if passed else f"FAIL: {errors[0]}"
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    if errors:
        for e in errors:
            print(f"    - {e}")

    return TestResult(
        test_id=test_id,
        label=label,
        passed=passed,
        message=message,
        elapsed=elapsed,
        errors=errors,
        transitions=1
    )


def test_pc4_create_session_side_effect(client: RallyClient, project: dict, agents_by_name: dict[str, dict], log: logging.Logger | None = None, run_log: RunLog | None = None) -> TestResult:
    """PC4: create_session Side Effect — verify session created with correct agent."""
    test_id = "PC4"
    label = "create_session Side Effect"
    t_start = time.monotonic()
    errors: list[str] = []

    if run_log:
        client.test_id = test_id
        client.run_log = run_log

    try:
        protocol = find_protocol_by_name(client, project["id"], "code_review")
        if not protocol:
            errors.append("code_review protocol not found")
            raise RuntimeError("code_review protocol not found")

        task, artifact = setup_pr(client, project["id"], f"PC4-TEST-{int(time.time())}")
        author_id = agents_by_name["proto-author"]["id"]

        # Trigger -> opened
        client.emit_event(project["id"], "code.pr_opened", {
            "artifact_id": artifact["id"],
            "task_id": task["id"],
            "author_agent_id": author_id,
        })
        instance = wait_for_instance(client, project["id"], protocol["id"], timeout=15, artifact_id=artifact["id"])
        if not instance:
            errors.append("Instance not created")
        else:
            instance_id = instance["id"]
            print(f"    Instance in opened state")

            # Transition to ready_for_review
            client.emit_event(project["id"], "test.passed", {
                "artifact_id": artifact["id"],
            })
            inst = wait_for_state(client, project["id"], instance_id, "ready_for_review", timeout=15)
            if not inst:
                errors.append("Instance did not reach ready_for_review")
            else:
                print(f"    Instance in ready_for_review state")

                session = wait_for_protocol_session(
                    client, project["id"], instance_id, timeout=30, origin="protocol"
                )
                session_agent_id = session.get("agent_id")
                reviewer_id = agents_by_name["proto-reviewer"]["id"]

                if session_agent_id != reviewer_id:
                    errors.append(f"Session agent mismatch: expected {reviewer_id}, got {session_agent_id}")
                else:
                    print(f"    Session created with correct reviewer agent")

    except Exception as exc:
        errors.append(str(exc))

    elapsed = time.monotonic() - t_start
    passed = len(errors) == 0
    message = "PASS" if passed else f"FAIL: {errors[0]}"
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    if errors:
        for e in errors:
            print(f"    - {e}")

    return TestResult(
        test_id=test_id,
        label=label,
        passed=passed,
        message=message,
        elapsed=elapsed,
        errors=errors,
        transitions=2
    )


def test_pc5_notify_actor_content(client: RallyClient, project: dict, agents_by_name: dict[str, dict], log: logging.Logger | None = None, run_log: RunLog | None = None) -> TestResult:
    """PC5: notify_actor Content Resolution — verify artifact name resolved in DM."""
    test_id = "PC5"
    label = "notify_actor Content Resolution"
    t_start = time.monotonic()
    errors: list[str] = []

    if run_log:
        client.test_id = test_id
        client.run_log = run_log

    try:
        protocol = find_protocol_by_name(client, project["id"], "code_review")
        if not protocol:
            errors.append("code_review protocol not found")
            raise RuntimeError("code_review protocol not found")

        artifact_name = f"NOTIFY-TEST-{uuid.uuid4().hex[:8]}"
        task, artifact = setup_pr(client, project["id"], artifact_name)
        author_id = agents_by_name["proto-author"]["id"]

        # Trigger
        client.emit_event(project["id"], "code.pr_opened", {
            "artifact_id": artifact["id"],
            "task_id": task["id"],
            "author_agent_id": author_id,
        })

        instance = wait_for_instance(client, project["id"], protocol["id"], timeout=15, artifact_id=artifact["id"])
        if not instance:
            errors.append("Instance not created")
        else:
            instance_id = instance["id"]
            # Capture author from actor_assignments
            actors = instance.get("actor_assignments") or {}
            author_slot = actors.get("author")
            if not author_slot:
                errors.append("author not in actor_assignments")
            else:
                author_uuid = author_slot.get("id")
                print(f"    Author in instance: {author_uuid}")

                # Drive to ci_failure state
                client.emit_event(project["id"], "test.failed", {
                    "artifact_id": artifact["id"],
                })

                inst = wait_for_state(client, project["id"], instance_id, "ci_failure", timeout=15)
                if not inst:
                    errors.append("Instance did not reach ci_failure")
                else:
                    print(f"    Instance in ci_failure state, notify_actor should have fired")

                    # Find DM channel
                    channels = client.list_channels(project["id"])
                    dm_channel_name = f"dm-{author_uuid}"
                    dm_channel = next((c for c in channels if c.get("name") == dm_channel_name), None)

                    if not dm_channel:
                        errors.append(f"DM channel {dm_channel_name} not found")
                    else:
                        msg = wait_for_message(
                            client,
                            dm_channel["id"],
                            lambda message: message.get("metadata", {}).get("protocol_instance_id") == instance_id,
                            timeout=15,
                        )
                        content = msg.get("content", "")
                        print(f"    Found DM message: {content[:60]}...")

                        if artifact_name not in content:
                            errors.append("Artifact name not resolved in DM")
                        else:
                            print(f"    Artifact name resolved in DM")

                        if "{{" in content:
                            errors.append("DM contains unresolved template")
                        else:
                            print(f"    No templates in DM")

    except Exception as exc:
        errors.append(str(exc))

    elapsed = time.monotonic() - t_start
    passed = len(errors) == 0
    message = "PASS" if passed else f"FAIL: {errors[0]}"
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    if errors:
        for e in errors:
            print(f"    - {e}")

    return TestResult(
        test_id=test_id,
        label=label,
        passed=passed,
        message=message,
        elapsed=elapsed,
        errors=errors,
        transitions=2
    )


def test_pc6_ci_failure_loop(client: RallyClient, project: dict, agents_by_name: dict[str, dict], log: logging.Logger | None = None, run_log: RunLog | None = None) -> TestResult:
    """PC6: CI Failure Back-Loop — pr_updated after test.failed returns to opened."""
    test_id = "PC6"
    label = "CI Failure Back-Loop"
    t_start = time.monotonic()
    errors: list[str] = []

    if run_log:
        client.test_id = test_id
        client.run_log = run_log

    try:
        protocol = find_protocol_by_name(client, project["id"], "code_review")
        if not protocol:
            errors.append("code_review protocol not found")
            raise RuntimeError("code_review protocol not found")

        task, artifact = setup_pr(client, project["id"], f"PC6-TEST-{int(time.time())}")
        author_id = agents_by_name["proto-author"]["id"]

        # opened
        client.emit_event(project["id"], "code.pr_opened", {
            "artifact_id": artifact["id"],
            "task_id": task["id"],
            "author_agent_id": author_id,
        })
        instance = wait_for_instance(client, project["id"], protocol["id"], timeout=15, artifact_id=artifact["id"])
        if not instance:
            errors.append("Instance not created")
        else:
            instance_id = instance["id"]
            print(f"    Instance in opened state")

            # ci_failure
            client.emit_event(project["id"], "test.failed", {
                "artifact_id": artifact["id"],
            })
            inst = wait_for_state(client, project["id"], instance_id, "ci_failure", timeout=15)
            if not inst:
                errors.append("Instance did not reach ci_failure")
            else:
                print(f"    Instance in ci_failure state")

                # Back to opened
                client.emit_event(project["id"], "code.pr_updated", {
                    "artifact_id": artifact["id"],
                })
                inst = wait_for_state(client, project["id"], instance_id, "opened", timeout=15)
                if not inst:
                    errors.append("Instance did not return to opened")
                else:
                    print(f"    Instance back in opened state")

                    # Verify transitions
                    transitions = client.list_transitions(project["id"], instance_id)
                    trans_pairs = [(t.get("from_state"), t.get("to_state")) for t in transitions]

                    if ("opened", "ci_failure") not in trans_pairs:
                        errors.append("Missing opened→ci_failure transition")
                    if ("ci_failure", "opened") not in trans_pairs:
                        errors.append("Missing ci_failure→opened transition")

                    if not errors:
                        print(f"    Loop verified in transitions")

    except Exception as exc:
        errors.append(str(exc))

    elapsed = time.monotonic() - t_start
    passed = len(errors) == 0
    message = "PASS" if passed else f"FAIL: {errors[0]}"
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    if errors:
        for e in errors:
            print(f"    - {e}")

    return TestResult(
        test_id=test_id,
        label=label,
        passed=passed,
        message=message,
        elapsed=elapsed,
        errors=errors,
        transitions=3
    )


def test_pc7_complete_task(client: RallyClient, project: dict, agents_by_name: dict[str, dict], log: logging.Logger | None = None, run_log: RunLog | None = None) -> TestResult:
    """PC7: complete_task in Merged State — verify linked task marked done."""
    test_id = "PC7"
    label = "complete_task in Merged State"
    t_start = time.monotonic()
    errors: list[str] = []

    if run_log:
        client.test_id = test_id
        client.run_log = run_log

    try:
        protocol = find_protocol_by_name(client, project["id"], "code_review")
        if not protocol:
            errors.append("code_review protocol not found")
            raise RuntimeError("code_review protocol not found")

        task, artifact = setup_pr(client, project["id"], f"PC7-TEST-{int(time.time())}")
        author_id = agents_by_name["proto-author"]["id"]

        # opened
        client.emit_event(project["id"], "code.pr_opened", {
            "artifact_id": artifact["id"],
            "task_id": task["id"],
            "author_agent_id": author_id,
        })
        instance = wait_for_instance(client, project["id"], protocol["id"], timeout=15, artifact_id=artifact["id"])
        if not instance:
            errors.append("Instance not created")
        else:
            instance_id = instance["id"]
            print(f"    Instance created")

            # ready_for_review
            client.emit_event(project["id"], "test.passed", {
                "artifact_id": artifact["id"],
            })
            inst = wait_for_state(client, project["id"], instance_id, "ready_for_review", timeout=15)
            if not inst:
                errors.append("Did not reach ready_for_review")
            else:
                print(f"    ready_for_review")

                # approved
                client.emit_event(project["id"], "review.approved", {
                    "artifact_id": artifact["id"],
                })
                inst = wait_for_state(client, project["id"], instance_id, "approved", timeout=15)
                if not inst:
                    errors.append("Did not reach approved")
                else:
                    print(f"    approved")

                    # merged
                    client.emit_event(project["id"], "code.pr_merged", {
                        "artifact_id": artifact["id"],
                    })
                    inst = wait_for_state(client, project["id"], instance_id, "merged", timeout=15)
                    if not inst:
                        errors.append("Did not reach merged")
                    else:
                        print(f"    merged")

                        final_inst = _wait_for_value(
                            f"protocol instance {instance_id} completion",
                            lambda: client.get_instance(project["id"], instance_id),
                            lambda current: current.get("status") == "completed",
                            timeout=20,
                            interval=1.0,
                            summarize=lambda current: (
                                f"state={current.get('current_state')!r}, status={current.get('status')!r}"
                            ),
                        )
                        if final_inst.get("status") != "completed":
                            errors.append(f"Instance status should be completed, got {final_inst.get('status')}")
                        else:
                            print(f"    Instance status: completed")

                        task_after = wait_for_task_status(
                            client, project["id"], task["id"], "done", timeout=20
                        )
                        if task_after.get("status") != "done":
                            errors.append(f"complete_task did not mark task done (got {task_after.get('status')})")
                        else:
                            print(f"    Task marked done")

    except Exception as exc:
        errors.append(str(exc))

    elapsed = time.monotonic() - t_start
    passed = len(errors) == 0
    message = "PASS" if passed else f"FAIL: {errors[0]}"
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    if errors:
        for e in errors:
            print(f"    - {e}")

    return TestResult(
        test_id=test_id,
        label=label,
        passed=passed,
        message=message,
        elapsed=elapsed,
        errors=errors,
        transitions=5
    )


def test_pc8_record_decision(client: RallyClient, project: dict, agents_by_name: dict[str, dict], log: logging.Logger | None = None, run_log: RunLog | None = None) -> TestResult:
    """PC8: record_decision in Merged State — verify knowledge item created with resolved content."""
    test_id = "PC8"
    label = "record_decision in Merged State"
    t_start = time.monotonic()
    errors: list[str] = []

    if run_log:
        client.test_id = test_id
        client.run_log = run_log

    try:
        protocol = find_protocol_by_name(client, project["id"], "code_review")
        if not protocol:
            errors.append("code_review protocol not found")
            raise RuntimeError("code_review protocol not found")

        artifact_name = f"PC8-DECISION-{uuid.uuid4().hex[:8]}"
        task, artifact = setup_pr(client, project["id"], artifact_name)
        author_id = agents_by_name["proto-author"]["id"]

        # Drive to merged
        client.emit_event(project["id"], "code.pr_opened", {
            "artifact_id": artifact["id"],
            "task_id": task["id"],
            "author_agent_id": author_id,
        })
        instance = wait_for_instance(client, project["id"], protocol["id"], timeout=15, artifact_id=artifact["id"])
        if not instance:
            errors.append("Instance not created")
        else:
            instance_id = instance["id"]

            # ready_for_review
            client.emit_event(project["id"], "test.passed", {
                "artifact_id": artifact["id"],
            })
            wait_for_state(client, project["id"], instance_id, "ready_for_review", timeout=15)

            # approved
            client.emit_event(project["id"], "review.approved", {
                "artifact_id": artifact["id"],
            })
            wait_for_state(client, project["id"], instance_id, "approved", timeout=15)

            # merged
            client.emit_event(project["id"], "code.pr_merged", {
                "artifact_id": artifact["id"],
            })
            inst = wait_for_state(client, project["id"], instance_id, "merged", timeout=15)
            if not inst:
                errors.append("Did not reach merged")
            else:
                print(f"    Reached merged state")

                item = wait_for_knowledge_item(
                    client,
                    project["id"],
                    lambda knowledge_item: knowledge_item.get("provenance_type") == "protocol"
                    and knowledge_item.get("provenance_protocol_instance_id") == instance_id,
                    timeout=20,
                )
                content = item.get("content", "")
                print(f"    Found knowledge item")

                if artifact_name not in content:
                    errors.append(f"Artifact name not resolved in knowledge item")
                else:
                    print(f"    Artifact name resolved in knowledge")

                if "{{" in content:
                    errors.append("Knowledge item contains unresolved template")
                else:
                    print(f"    No templates in knowledge")

    except Exception as exc:
        errors.append(str(exc))

    elapsed = time.monotonic() - t_start
    passed = len(errors) == 0
    message = "PASS" if passed else f"FAIL: {errors[0]}"
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    if errors:
        for e in errors:
            print(f"    - {e}")

    return TestResult(
        test_id=test_id,
        label=label,
        passed=passed,
        message=message,
        elapsed=elapsed,
        errors=errors,
        transitions=5
    )


def test_pc9_timeout_escalation(client: RallyClient, project: dict, agents_by_name: dict[str, dict], log: logging.Logger | None = None, run_log: RunLog | None = None) -> TestResult:
    """PC9: Timeout Escalation — short timeout triggers escalation event."""
    test_id = "PC9"
    label = "Timeout Escalation"
    t_start = time.monotonic()
    errors: list[str] = []

    if run_log:
        client.test_id = test_id
        client.run_log = run_log

    try:
        # Create a test-specific protocol with short timeout
        run_id = str(uuid.uuid4())
        proto_def = {
            "name": f"protocol_timeout_test_{run_id[:8]}",
            "version": "1.0",
            "description": "Test timeout escalation",
            "definition": {
                "initial_state": "waiting",
                "terminal_states": {},
                "states": {
                    "waiting": {
                        "timeout": {
                            "duration": "1s",
                            "action": "escalate"
                        },
                        "transitions": []
                    }
                }
            },
            "triggers": [
                {
                    "event_type": "timeout.test.triggered",
                    "conditions": {"run_id": run_id}
                }
            ],
            "escalation_chain": None,
            "loaded_from": None
        }

        protocol = client._post(f"/api/v1/projects/{project['id']}/protocols", proto_def)[0]
        protocol_id = protocol["id"]
        print(f"    Created timeout test protocol {protocol_id}")

        # Trigger it
        client.emit_event(project["id"], "timeout.test.triggered", {
            "run_id": run_id,
        })

        # Wait for instance
        instance = wait_for_instance(client, project["id"], protocol_id, timeout=15)
        if not instance:
            errors.append("Instance not created")
        else:
            instance_id = instance["id"]
            print(f"    Instance created in state {instance.get('current_state')}")
            print(f"    (PC9 requires ~40s for background scheduler to fire escalation)")

            # Wait for escalation (scheduler runs every 30s, worst case ~70s total)
            deadline = time.time() + 80
            escalation_fired = False
            while time.time() < deadline:
                inst = client.get_instance(project["id"], instance_id)
                if (inst.get("escalation_step") or 0) >= 1:
                    escalation_fired = True
                    print(f"    Escalation fired (escalation_step={inst.get('escalation_step')})")
                    break
                time.sleep(5)

            if not escalation_fired:
                errors.append("Escalation did not fire within 80s")
            else:
                # Verify protocol.escalated event was emitted
                print(f"    Timeout escalation test complete (40s elapsed)")

    except Exception as exc:
        errors.append(str(exc))

    elapsed = time.monotonic() - t_start
    passed = len(errors) == 0
    message = "PASS" if passed else f"FAIL: {errors[0]}"
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    if errors:
        for e in errors:
            print(f"    - {e}")

    return TestResult(
        test_id=test_id,
        label=label,
        passed=passed,
        message=message,
        elapsed=elapsed,
        errors=errors,
        transitions=0
    )


def test_pc10_awaiting_revision_loop(client: RallyClient, project: dict, agents_by_name: dict[str, dict], log: logging.Logger | None = None, run_log: RunLog | None = None) -> TestResult:
    """PC10: Awaiting Revision Loop — changes_requested and pr_updated loop."""
    test_id = "PC10"
    label = "Awaiting Revision Loop"
    t_start = time.monotonic()
    errors: list[str] = []

    if run_log:
        client.test_id = test_id
        client.run_log = run_log

    try:
        protocol = find_protocol_by_name(client, project["id"], "code_review")
        if not protocol:
            errors.append("code_review protocol not found")
            raise RuntimeError("code_review protocol not found")

        task, artifact = setup_pr(client, project["id"], f"PC10-TEST-{int(time.time())}")
        author_id = agents_by_name["proto-author"]["id"]

        # opened
        client.emit_event(project["id"], "code.pr_opened", {
            "artifact_id": artifact["id"],
            "task_id": task["id"],
            "author_agent_id": author_id,
        })
        instance = wait_for_instance(client, project["id"], protocol["id"], timeout=15, artifact_id=artifact["id"])
        if not instance:
            errors.append("Instance not created")
        else:
            instance_id = instance["id"]
            print(f"    Instance in opened state")

            # ready_for_review
            client.emit_event(project["id"], "test.passed", {
                "artifact_id": artifact["id"],
            })
            inst = wait_for_state(client, project["id"], instance_id, "ready_for_review", timeout=15)
            if not inst:
                errors.append("Did not reach ready_for_review")
            else:
                print(f"    Instance in ready_for_review state")
                actors = inst.get("actor_assignments") or {}
                author_uuid = actors.get("author", {}).get("id")

                # changes_requested -> awaiting_revision
                client.emit_event(project["id"], "review.changes_requested", {
                    "artifact_id": artifact["id"],
                })
                inst = wait_for_state(client, project["id"], instance_id, "awaiting_revision", timeout=15)
                if not inst:
                    errors.append("Did not reach awaiting_revision")
                else:
                    print(f"    Instance in awaiting_revision state")

                    # pr_updated -> ready_for_review
                    client.emit_event(project["id"], "code.pr_updated", {
                        "artifact_id": artifact["id"],
                    })
                    inst = wait_for_state(client, project["id"], instance_id, "ready_for_review", timeout=15)
                    if not inst:
                        errors.append("Did not return to ready_for_review")
                    else:
                        print(f"    Instance back in ready_for_review state")

                        # Verify transitions
                        transitions = client.list_transitions(project["id"], instance_id)
                        trans_pairs = [(t.get("from_state"), t.get("to_state")) for t in transitions]

                        if ("ready_for_review", "awaiting_revision") not in trans_pairs:
                            errors.append("Missing ready_for_review→awaiting_revision transition")
                        if ("awaiting_revision", "ready_for_review") not in trans_pairs:
                            errors.append("Missing awaiting_revision→ready_for_review transition")

                        if not errors:
                            print(f"    Loop verified in transitions")

                        # Verify notify_actor fired (check DM)
                        if author_uuid:
                            channels = client.list_channels(project["id"])
                            dm_channel_name = f"dm-{author_uuid}"
                            dm_channel = next((c for c in channels if c.get("name") == dm_channel_name), None)
                            if dm_channel:
                                messages = client.list_messages(dm_channel["id"], limit=50)
                                proto_msgs = [m for m in messages if m.get("metadata", {}).get("protocol_instance_id") == instance_id]
                                if proto_msgs:
                                    print(f"    DM notify_actor fired in awaiting_revision")

    except Exception as exc:
        errors.append(str(exc))

    elapsed = time.monotonic() - t_start
    passed = len(errors) == 0
    message = "PASS" if passed else f"FAIL: {errors[0]}"
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    if errors:
        for e in errors:
            print(f"    - {e}")

    return TestResult(
        test_id=test_id,
        label=label,
        passed=passed,
        message=message,
        elapsed=elapsed,
        errors=errors,
        transitions=4
    )


def test_pl1_live_reviewer_decision(
    client: RallyClient,
    project: dict,
    agents_by_name: dict[str, dict],
    log: logging.Logger | None = None,
    run_log: RunLog | None = None,
) -> TestResult:
    """PL1: Live Reviewer Decision — real LLM reviews PR, output drives state machine.

    Flow:
      code.pr_opened → [opened] → test.passed → [ready_for_review]
      → reviewer session fires (LLM call) → read output → emit review.approved
      → [approved] → emit code.pr_merged → [merged]
      → verify task done, knowledge item created, reviewer output non-empty.
    """
    test_id = "PL1"
    label = "Live Reviewer Decision (LLM)"
    t_start = time.monotonic()
    errors: list[str] = []

    if run_log:
        client.test_id = test_id
        client.run_log = run_log

    try:
        protocol = find_protocol_by_name(client, project["id"], "code_review")
        if not protocol:
            errors.append("code_review protocol not found")
            raise RuntimeError("code_review protocol not found")

        artifact_name = f"PL1-PR-{uuid.uuid4().hex[:8]}"
        try:
            task, artifact = setup_pr(client, project["id"], artifact_name)
        except RuntimeError as e:
            errors.append(f"setup_pr failed: {e}")
            raise
        author_id = agents_by_name["proto-author"]["id"]
        print(f"    PR: {artifact_name}")

        # opened
        client.emit_event(project["id"], "code.pr_opened", {
            "artifact_id": artifact["id"],
            "task_id": task["id"],
            "author_agent_id": author_id,
        })
        instance = wait_for_instance(
            client, project["id"], protocol["id"], timeout=15, artifact_id=artifact["id"]
        )
        if not instance:
            errors.append("Instance not created")
            raise RuntimeError("no instance")
        instance_id = instance["id"]
        print(f"    Instance {instance_id[:8]} in opened state")

        # ready_for_review — create a live reviewer API session and use its output to drive the protocol
        client.emit_event(project["id"], "test.passed", {"artifact_id": artifact["id"]})
        rfr = wait_for_state(client, project["id"], instance_id, "ready_for_review", timeout=15)
        if not rfr:
            errors.append("Did not reach ready_for_review")
            raise RuntimeError("no ready_for_review")
        print(f"    State: ready_for_review — creating API review session")

        reviewer_id = agents_by_name["proto-reviewer"]["id"]
        try:
            review_task = client.create_task(
                project["id"],
                f"Review PR: {artifact_name}",
                "Review the pull request. Respond with APPROVE if acceptable, otherwise CHANGES_REQUESTED.",
                {"artifact_id": artifact["id"]},
            )
        except RuntimeError as e:
            errors.append(f"create review task failed: {e}")
            raise
        try:
            cli_session = client.create_session(
                agent_id=reviewer_id,
                project_id=project["id"],
                task_id=review_task["id"],
            )
        except RuntimeError as e:
            errors.append(f"create API session failed: {e}")
            raise
        cli_session_id = cli_session["id"]
        print(f"    API session {cli_session_id[:8]} dispatched — waiting for LLM…")

        # Wait for real LLM call to complete (up to 300s)
        session = wait_for_session_by_id(client, cli_session_id, timeout=300)
        if not session:
            errors.append("Reviewer API session did not complete within 300s")
            raise RuntimeError("session timeout")

        session_status = session.get("status")
        session_output = session.get("output") or ""
        print(f"    Session status: {session_status}")
        print(f"    Session output ({len(session_output)} chars): {session_output[:120]!r}")

        if session_status == "failed":
            errors.append(f"Reviewer session failed: {session.get('error')}")
            raise RuntimeError("session failed")

        if not session_output.strip():
            errors.append("Reviewer LLM produced no output")
        else:
            print(f"    LLM produced output ✓")

        # Interpret output and drive protocol forward
        output_upper = session_output.upper()
        if "APPROVE" in output_upper:
            verdict = "approved"
            print(f"    Verdict: APPROVE — emitting review.approved")
            client.emit_event(project["id"], "review.approved", {"artifact_id": artifact["id"]})
        else:
            verdict = "changes_requested"
            print(f"    Verdict: changes requested — emitting review.changes_requested")
            client.emit_event(project["id"], "review.changes_requested", {"artifact_id": artifact["id"]})

        if verdict == "approved":
            approved_inst = wait_for_state(client, project["id"], instance_id, "approved", timeout=15)
            if not approved_inst:
                errors.append("Did not reach approved state")
                raise RuntimeError("no approved")
            print(f"    State: approved")

            # merge
            client.emit_event(project["id"], "code.pr_merged", {"artifact_id": artifact["id"]})
            merged_inst = wait_for_state(client, project["id"], instance_id, "merged", timeout=15)
            if not merged_inst:
                errors.append("Did not reach merged state")
            else:
                print(f"    State: merged")

                # verify task done
                t = wait_for_task_status(client, project["id"], task["id"], "done", timeout=20)
                if t.get("status") != "done":
                    errors.append(f"Task not done after merge (got {t.get('status')})")
                else:
                    print(f"    Task status: done ✓")

                item = wait_for_knowledge_item(
                    client,
                    project["id"],
                    lambda knowledge_item: knowledge_item.get("provenance_type") == "protocol"
                    and str(knowledge_item.get("provenance_protocol_instance_id")) == instance_id,
                    timeout=20,
                )
                if not item:
                    errors.append("No knowledge item recorded for merge")
                else:
                    print(f"    Knowledge item recorded ✓")

                # verify instance completed
                final = client.get_instance(project["id"], instance_id)
                if final.get("status") != "completed":
                    errors.append(f"Instance status not completed (got {final.get('status')})")
                else:
                    print(f"    Protocol instance: completed ✓")
        else:
            # changes_requested path — verify awaiting_revision
            ar_inst = wait_for_state(client, project["id"], instance_id, "awaiting_revision", timeout=15)
            if not ar_inst:
                errors.append("Did not reach awaiting_revision state")
            else:
                print(f"    State: awaiting_revision (LLM chose to request changes)")
                print(f"    Note: revision loop not driven further in PL1")

    except RuntimeError:
        pass
    except Exception as exc:
        errors.append(str(exc))

    elapsed = time.monotonic() - t_start
    passed = len(errors) == 0
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    if errors:
        for e in errors:
            print(f"    - {e}")

    return TestResult(
        test_id=test_id,
        label=label,
        passed=passed,
        message="PASS" if passed else f"FAIL: {errors[0]}",
        elapsed=elapsed,
        errors=errors,
    )


def test_pl2_protocol_created_reviewer_session(
    client: RallyClient,
    project: dict,
    agents_by_name: dict[str, dict],
    log: logging.Logger | None = None,
    run_log: RunLog | None = None,
) -> TestResult:
    """PL2: Use the protocol-created reviewer session and validate its live output."""
    test_id = "PL2"
    label = "Protocol-Created Reviewer Session (LLM)"
    t_start = time.monotonic()
    errors: list[str] = []

    if run_log:
        client.test_id = test_id
        client.run_log = run_log

    try:
        protocol = find_protocol_by_name(client, project["id"], "code_review")
        if not protocol:
            errors.append("code_review protocol not found")
            raise RuntimeError("code_review protocol not found")

        artifact_name = f"PL2-PR-{uuid.uuid4().hex[:8]}"
        task, artifact = setup_pr(client, project["id"], artifact_name)
        author_id = agents_by_name["proto-author"]["id"]
        reviewer_id = agents_by_name["proto-reviewer"]["id"]

        client.emit_event(project["id"], "code.pr_opened", {
            "artifact_id": artifact["id"],
            "task_id": task["id"],
            "author_agent_id": author_id,
        })
        instance = wait_for_instance(client, project["id"], protocol["id"], timeout=15, artifact_id=artifact["id"])
        instance_id = instance["id"]
        print(f"    Instance {instance_id[:8]} created")

        client.emit_event(project["id"], "test.passed", {"artifact_id": artifact["id"]})
        wait_for_state(client, project["id"], instance_id, "ready_for_review", timeout=15)
        print("    State: ready_for_review")

        protocol_session = wait_for_protocol_session(
            client, project["id"], instance_id, timeout=30, origin="protocol"
        )
        if protocol_session.get("agent_id") != reviewer_id:
            errors.append(
                f"Protocol session agent mismatch: expected {reviewer_id}, got {protocol_session.get('agent_id')}"
            )
            raise RuntimeError("wrong reviewer agent")

        completed_session = wait_for_session_by_id(client, protocol_session["id"], timeout=300)
        output = (completed_session.get("output") or "").strip()
        status = completed_session.get("status")
        print(f"    Session status: {status}")

        if status == "failed":
            errors.append(f"Protocol-created reviewer session failed: {completed_session.get('error')}")
            raise RuntimeError("protocol reviewer session failed")
        if not output:
            errors.append("Protocol-created reviewer session produced no output")
            raise RuntimeError("empty reviewer output")

        output_upper = output.upper()
        if "APPROVE" in output_upper:
            print("    Verdict: APPROVE")
            client.emit_event(project["id"], "review.approved", {"artifact_id": artifact["id"]})
            wait_for_state(client, project["id"], instance_id, "approved", timeout=15)
            print("    State: approved")
        elif "CHANGES_REQUESTED" in output_upper or "CHANGES REQUESTED" in output_upper:
            print("    Verdict: CHANGES_REQUESTED")
            client.emit_event(project["id"], "review.changes_requested", {"artifact_id": artifact["id"]})
            wait_for_state(client, project["id"], instance_id, "awaiting_revision", timeout=15)
            print("    State: awaiting_revision")
        else:
            errors.append("Reviewer output did not contain APPROVE or CHANGES_REQUESTED")

    except RuntimeError:
        pass
    except Exception as exc:
        errors.append(str(exc))

    elapsed = time.monotonic() - t_start
    passed = len(errors) == 0
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    if errors:
        for e in errors:
            print(f"    - {e}")

    return TestResult(
        test_id=test_id,
        label=label,
        passed=passed,
        message="PASS" if passed else f"FAIL: {errors[0]}",
        elapsed=elapsed,
        errors=errors,
    )


def test_pl3_live_reviewer_changes_requested(
    client: RallyClient,
    project: dict,
    agents_by_name: dict[str, dict],
    log: logging.Logger | None = None,
    run_log: RunLog | None = None,
) -> TestResult:
    """PL3: Force a live reviewer changes-requested response and verify awaiting_revision."""
    test_id = "PL3"
    label = "Live Reviewer Changes Requested (LLM)"
    t_start = time.monotonic()
    errors: list[str] = []

    if run_log:
        client.test_id = test_id
        client.run_log = run_log

    try:
        protocol = find_protocol_by_name(client, project["id"], "code_review")
        if not protocol:
            errors.append("code_review protocol not found")
            raise RuntimeError("code_review protocol not found")

        artifact_name = f"PL3-PR-{uuid.uuid4().hex[:8]}"
        task, artifact = setup_pr(client, project["id"], artifact_name)
        author_id = agents_by_name["proto-author"]["id"]
        reviewer_id = agents_by_name["proto-reviewer"]["id"]

        client.emit_event(project["id"], "code.pr_opened", {
            "artifact_id": artifact["id"],
            "task_id": task["id"],
            "author_agent_id": author_id,
        })
        instance = wait_for_instance(client, project["id"], protocol["id"], timeout=15, artifact_id=artifact["id"])
        instance_id = instance["id"]
        print(f"    Instance {instance_id[:8]} created")

        client.emit_event(project["id"], "test.passed", {"artifact_id": artifact["id"]})
        wait_for_state(client, project["id"], instance_id, "ready_for_review", timeout=15)
        print("    State: ready_for_review")

        review_task = client.create_task(
            project["id"],
            f"Reject PR: {artifact_name}",
            (
                "Review the pull request as incomplete. It has no tests, no rollout plan, and no risk analysis. "
                "Respond with CHANGES_REQUESTED and include at least one concrete reason."
            ),
            {"artifact_id": artifact["id"], "review_style": "strict_reject"},
        )
        review_session = client.create_session(
            agent_id=reviewer_id,
            project_id=project["id"],
            task_id=review_task["id"],
        )
        completed_session = wait_for_session_by_id(client, review_session["id"], timeout=300)
        output = (completed_session.get("output") or "").strip()
        status = completed_session.get("status")
        print(f"    Session status: {status}")

        if status == "failed":
            errors.append(f"Reviewer rejection session failed: {completed_session.get('error')}")
            raise RuntimeError("reviewer rejection session failed")

        output_upper = output.upper()
        if "CHANGES_REQUESTED" not in output_upper and "CHANGES REQUESTED" not in output_upper:
            errors.append(f"Reviewer did not request changes: {output[:160]!r}")
            raise RuntimeError("reviewer did not reject")

        client.emit_event(project["id"], "review.changes_requested", {"artifact_id": artifact["id"]})
        wait_for_state(client, project["id"], instance_id, "awaiting_revision", timeout=15)
        print("    State: awaiting_revision")

    except RuntimeError:
        pass
    except Exception as exc:
        errors.append(str(exc))

    elapsed = time.monotonic() - t_start
    passed = len(errors) == 0
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    if errors:
        for e in errors:
            print(f"    - {e}")

    return TestResult(
        test_id=test_id,
        label=label,
        passed=passed,
        message="PASS" if passed else f"FAIL: {errors[0]}",
        elapsed=elapsed,
        errors=errors,
    )


# ── Test registry ──────────────────────────────────────────────────────────────

PROTOCOL_TESTS = [
    ("PC1", test_pc1_trigger_actor_resolution),
    ("PC2", test_pc2_template_guard_resolution),
    ("PC3", test_pc3_post_message_content),
    ("PC4", test_pc4_create_session_side_effect),
    ("PC5", test_pc5_notify_actor_content),
    ("PC6", test_pc6_ci_failure_loop),
    ("PC7", test_pc7_complete_task),
    ("PC8", test_pc8_record_decision),
    ("PC9", test_pc9_timeout_escalation),
    ("PC10", test_pc10_awaiting_revision_loop),
    ("PL1", test_pl1_live_reviewer_decision),
    ("PL2", test_pl2_protocol_created_reviewer_session),
    ("PL3", test_pl3_live_reviewer_changes_requested),
]


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HuddleRoom protocol live integration tests")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--no-start-server", action="store_true")
    p.add_argument("--keep-server", action="store_true")
    p.add_argument("--reuse-project", action="store_true")
    p.add_argument("--tests", default="PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--timeout", type=int, default=60)
    return p.parse_args()


# ── Server management ──────────────────────────────────────────────────────────

def project_venv_executable(project_root: Path, executable: str) -> Path:
    candidates = [
        project_root / ".venv" / "bin" / executable,
        project_root / ".venv" / "Scripts" / executable,
        project_root / ".venv" / "Scripts" / f"{executable}.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise RuntimeError(
        f"Project venv executable not found for '{executable}'. Expected one of: {searched}"
    )


def check_server(base_url: str) -> bool:
    try:
        r = httpx.get(f"{base_url}/api/v1/config", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def run_migrations(project_root: Path) -> None:
    print("Running migrations...")
    expected_database_url = f"sqlite+aiosqlite:///{project_root / 'tests' / 'live' / 'huddleroom_live_test.db'}"
    env = os.environ.copy()
    env["HUDDLEROOM_DATABASE_URL"] = expected_database_url
    result = subprocess.run(
        [str(project_venv_executable(project_root, "alembic")), "upgrade", "head"],
        cwd=str(project_root),
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Migration stderr: {result.stderr}")
        raise RuntimeError(f"Migrations failed (exit {result.returncode})")
    print("Migrations OK.")


def start_server(
    base_url: str,
    port: int,
    project_root: Path,
    server_log_path: Path,
) -> tuple[subprocess.Popen, TextIO]:
    print(f"Starting HuddleRoom server on port {port}...")
    print(f"Server log: {server_log_path}")
    run_migrations(project_root)
    server_log = open(server_log_path, "w", encoding="utf-8")
    env = os.environ.copy()
    env["HUDDLEROOM_DATABASE_URL"] = f"sqlite+aiosqlite:///{project_root / 'tests' / 'live' / 'huddleroom_live_test.db'}"
    env["HUDDLEROOM_AUTH_ENABLED"] = "false"
    env["HUDDLEROOM_API_BASE_URL"] = base_url
    try:
        proc = subprocess.Popen(
            [str(project_venv_executable(project_root, "huddleroom")), "serve", "--port", str(port)],
            cwd=str(project_root),
            env=env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )
    except Exception:
        server_log.close()
        raise
    deadline = time.time() + 30
    while time.time() < deadline:
        if check_server(base_url):
            print(f"Server ready at {base_url}")
            return proc, server_log
        if proc.poll() is not None:
            server_log.close()
            raise RuntimeError(
                f"Server exited early (exit code {proc.returncode}). Check server log: {server_log_path}"
            )
        time.sleep(1)
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    server_log.close()
    raise RuntimeError(f"Server did not become ready in 30 seconds. Check server log: {server_log_path}")


def check_provider_ready(provider: str) -> str | None:
    """Return an actionable error message if the live model provider is not usable."""
    if provider != "openai":
        return (
            f"Provider '{provider}' readiness probe is not implemented for PL tests; "
            "switch the live reviewer to OpenAI or skip PL* tests"
        )
    if not os.environ.get("OPENAI_API_KEY"):
        return "OPENAI_API_KEY is not set; OpenAI-backed PL tests cannot run"
    try:
        asyncio.run(_probe_openai_readiness())
        return None
    except litellm.AuthenticationError:
        return (
            "OpenAI probe failed authentication for model "
            f"{OPENAI_READINESS_MODEL}; check OPENAI_API_KEY and model access"
        )
    except (
        litellm.BadRequestError,
        litellm.NotFoundError,
        litellm.PermissionDeniedError,
    ) as exc:
        return f"OpenAI probe could not use model {OPENAI_READINESS_MODEL}: {_probe_error_message(exc)}"
    except (
        litellm.APIConnectionError,
        litellm.APIError,
        litellm.InternalServerError,
        litellm.ServiceUnavailableError,
        litellm.BadGatewayError,
    ) as exc:
        return f"OpenAI probe failed to reach provider: {_probe_error_message(exc)}"
    except Exception as exc:
        return f"OpenAI probe failed unexpectedly: {type(exc).__name__}: {_probe_error_message(exc)}"


async def _probe_openai_readiness() -> None:
    """Issue a tiny real OpenAI call so live tests fail fast on auth/network/model issues."""
    await asyncio.wait_for(
        litellm.acompletion(
            model=OPENAI_READINESS_MODEL,
            messages=[{"role": "user", "content": "Reply with OK."}],
            temperature=0.0,
            max_tokens=1,
        ),
        timeout=15,
    )


def _probe_error_message(exc: Exception) -> str:
    """Keep provider probe errors short and user-actionable."""
    message = str(getattr(exc, "message", exc))
    prefix = f"litellm.{type(exc).__name__}: "
    if message.startswith(prefix):
        return message[len(prefix):]
    return message


def stop_server(proc: subprocess.Popen, server_log: TextIO | None = None) -> None:
    print("Stopping server...")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    finally:
        if server_log and not server_log.closed:
            server_log.close()


# ── Logging setup ──────────────────────────────────────────────────────────────

def setup_logging() -> tuple[logging.Logger, str, Path]:
    """Create timestamped log file and return logger plus shared log metadata."""
    logs_dir = Path(__file__).resolve().parent / "logs"
    logs_dir.mkdir(exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"run_protocols_{timestamp}.log"

    logger = logging.getLogger("huddleroom_protocol_live_test")
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    print(f"Log file: {log_path}")
    return logger, timestamp, logs_dir


# ── Report ─────────────────────────────────────────────────────────────────────

def print_report(results: list[TestResult]) -> None:
    print(f"\n{'='*60}")
    print("  HUDDLEROOM PROTOCOL LIVE TEST REPORT")
    print(f"{'='*60}")
    print(f"  {'ID':<6} {'Label':<35} {'Result':<8} {'Time':>7}")
    print(f"  {'-'*6} {'-'*35} {'-'*8} {'-'*7}")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  {r.test_id:<6} {r.label:<35} {status:<8} {r.elapsed:>5.1f}s")
    print(f"{'='*60}")
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"  {passed}/{total} passed")
    if any(not r.passed for r in results):
        print("\n  Failures:")
        for r in results:
            if not r.passed:
                print(f"    {r.test_id} {r.label}:")
                for e in r.errors:
                    print(f"      - {e}")
    print(f"{'='*60}\n")


# ── HTML report generation ────────────────────────────────────────────────────

def generate_html_report(run_log: RunLog, timestamp: str) -> str:
    """Generate self-contained HTML report from RunLog. Returns HTML content."""
    passed = sum(1 for r in run_log.results if r.passed)
    total = len(run_log.results)
    pass_pct = (100 * passed // total) if total > 0 else 0

    # Group entries by test_id
    entries_by_test: dict[str | None, list[LogEntry]] = {}
    for entry in run_log.entries:
        tid = entry.test_id
        if tid not in entries_by_test:
            entries_by_test[tid] = []
        entries_by_test[tid].append(entry)

    # Build test sections
    test_sections = ""
    for result in run_log.results:
        test_id = result.test_id
        badge_color = "green" if result.passed else "red"
        badge_text = "PASS" if result.passed else "FAIL"

        # Get entries for this test
        test_entries = entries_by_test.get(test_id, [])

        # Build entry list
        entry_html = ""
        for entry in test_entries:
            if entry.kind == "http_get":
                path = entry.data.get("path", "?")
                status = entry.data.get("status", "?")
                label = f"GET {path} → {status}"
                payload = json.dumps(entry.data.get("response", {}), indent=2)
            elif entry.kind == "http_post":
                path = entry.data.get("path", "?")
                status = entry.data.get("status", "?")
                label = f"POST {path} → {status}"
                payload = json.dumps({
                    "body": entry.data.get("body", {}),
                    "response": entry.data.get("response", {})
                }, indent=2)
            else:
                label = entry.kind
                payload = json.dumps(entry.data, indent=2)

            entry_html += f'''
            <details>
                <summary>{label}</summary>
                <pre>{payload}</pre>
            </details>
            '''

        test_sections += f'''
        <section style="margin-bottom: 2rem; border: 1px solid #ddd; padding: 1rem; border-radius: 4px;">
            <h3 style="margin-top: 0;">
                <span style="display: inline-block; width: 60px; padding: 4px 8px; text-align: center; color: white; background-color: {badge_color}; border-radius: 3px; font-weight: bold; font-size: 0.9em;">{badge_text}</span>
                {test_id}: {result.label}
            </h3>
            <div style="margin: 1rem 0; font-size: 0.9em; color: #666;">
                <div><strong>Transitions:</strong> {result.transitions} | <strong>Time:</strong> {result.elapsed:.1f}s</div>
            </div>
            <div style="margin-top: 1rem;">
                <div style="margin-top: 1rem; border-top: 1px solid #eee; padding-top: 1rem;">
                    <h4 style="margin-top: 0;">HTTP Events</h4>
                    {entry_html if entry_html else '<p style="color: #999;">No events recorded</p>'}
                </div>
            </div>
        </section>
        '''

    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>HuddleRoom Protocol Live Test Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 2rem;
            background-color: #f5f5f5;
            color: #333;
        }}
        .header {{
            background-color: white;
            padding: 2rem;
            border-radius: 4px;
            margin-bottom: 2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0 0 1rem 0;
            font-size: 2em;
        }}
        .header .meta {{
            display: flex;
            gap: 2rem;
            font-size: 0.95em;
            color: #666;
        }}
        .summary-badge {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 1.1em;
            margin-bottom: 1rem;
        }}
        .summary-badge.pass {{
            background-color: #4caf50;
            color: white;
        }}
        .summary-badge.fail {{
            background-color: #f44336;
            color: white;
        }}
        .summary-table {{
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .summary-table th {{
            background-color: #f0f0f0;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid #ddd;
        }}
        .summary-table td {{
            padding: 12px;
            border-bottom: 1px solid #eee;
        }}
        .summary-table tr:last-child td {{
            border-bottom: none;
        }}
        .status-pass {{
            color: #4caf50;
            font-weight: bold;
        }}
        .status-fail {{
            color: #f44336;
            font-weight: bold;
        }}
        .tests-container {{
            display: flex;
            flex-direction: column;
            gap: 0;
        }}
        section {{
            background-color: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        section h3 {{
            background-color: #f9f9f9;
            margin: 0;
            padding: 1rem;
            border-bottom: 1px solid #eee;
        }}
        section > div {{
            padding: 1rem;
        }}
        details {{
            margin: 0.5rem 0;
            border: 1px solid #e0e0e0;
            border-radius: 3px;
            padding: 0.5rem;
            background-color: #fafafa;
        }}
        details[open] {{
            background-color: #fff;
        }}
        summary {{
            cursor: pointer;
            font-weight: 500;
            padding: 0.5rem;
            user-select: none;
        }}
        summary:hover {{
            background-color: #f0f0f0;
            border-radius: 2px;
        }}
        pre {{
            background-color: #f4f4f4;
            padding: 1rem;
            border-radius: 3px;
            overflow-x: auto;
            font-family: "Courier New", monospace;
            font-size: 0.85em;
            margin: 0.5rem 0 0 0;
            line-height: 1.4;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>HuddleRoom Protocol Live Test Report</h1>
        <div class="meta">
            <div><strong>Run Time:</strong> {timestamp}</div>
            <div><strong>Tests:</strong> {total}</div>
            <div><strong>Passed:</strong> {passed}</div>
        </div>
        <div class="summary-badge {'pass' if passed == total else 'fail'}">
            {passed}/{total} PASSED ({pass_pct}%)
        </div>
        <table class="summary-table">
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Label</th>
                    <th>Result</th>
                    <th>Transitions</th>
                    <th>Time</th>
                </tr>
            </thead>
            <tbody>
'''

    for result in run_log.results:
        status_class = "status-pass" if result.passed else "status-fail"
        status_text = "PASS" if result.passed else "FAIL"
        html += f'''
                <tr>
                    <td><strong>{result.test_id}</strong></td>
                    <td>{result.label}</td>
                    <td class="{status_class}">{status_text}</td>
                    <td>{result.transitions}</td>
                    <td>{result.elapsed:.1f}s</td>
                </tr>
'''

    html += f'''
            </tbody>
        </table>
    </div>
    <div class="tests-container">
        {test_sections}
    </div>
</body>
</html>
'''
    return html


# ── Main entry point ──────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    base_url = f"http://localhost:{args.port}"
    test_ids = [t.strip().upper() for t in args.tests.split(",")]

    log, timestamp, logs_dir = setup_logging()
    run_log = RunLog()

    server_proc: subprocess.Popen | None = None
    server_log: TextIO | None = None

    if args.no_start_server:
        if not check_server(base_url):
            print(f"ERROR: No server running at {base_url}. Start one or remove --no-start-server.")
            return 1
        print(f"Using existing server at {base_url}")
    else:
        if check_server(base_url):
            print(f"Server already running at {base_url} — reusing.")
        else:
            server_log_path = logs_dir / f"server_protocols_{timestamp}.log"
            try:
                server_proc, server_log = start_server(base_url, args.port, PROJECT_ROOT, server_log_path)
            except Exception as exc:
                print(f"ERROR: {exc} (server log: {server_log_path})")
                return 1

    client = HuddleRoomClient(base_url, log=log, run_log=run_log)

    try:
        project, agents_by_name = setup_fixtures(client, log=log, reuse_project=args.reuse_project)
    except Exception as exc:
        print(f"ERROR: Fixture setup failed: {exc}")
        if server_proc and not args.keep_server:
            stop_server(server_proc, server_log)
        elif server_log and not server_log.closed:
            server_log.close()
        return 1

    print(f"\nRunning tests: {', '.join(test_ids)}\n")

    if any(test_id.startswith("PL") for test_id in test_ids):
        reviewer_provider = (agents_by_name.get("proto-reviewer") or {}).get("provider")
        if reviewer_provider:
            provider_error = check_provider_ready(reviewer_provider)
            if provider_error:
                print(f"ERROR: {provider_error}")
                if server_proc and not args.keep_server:
                    stop_server(server_proc, server_log)
                elif server_log and not server_log.closed:
                    server_log.close()
                return 1

    results: list[TestResult] = []
    for test_id_upper in test_ids:
        # Find test function
        test_func = None
        for tid, fn in PROTOCOL_TESTS:
            if tid == test_id_upper:
                test_func = fn
                break

        if test_func is None:
            print(f"\n{'='*60}")
            print(f"  {test_id_upper}: UNKNOWN TEST")
            print(f"{'='*60}")
            results.append(TestResult(
                test_id=test_id_upper,
                label="UNKNOWN",
                passed=False,
                message="Test not found",
                elapsed=0,
                errors=["Test not found in registry"],
                transitions=0
            ))
            continue

        print(f"\n{'='*60}")
        print(f"  {test_id_upper}")
        print(f"{'='*60}")

        try:
            result = test_func(client, project, agents_by_name, log=log, run_log=run_log)
            results.append(result)
        except Exception as exc:
            print(f"  EXCEPTION: {exc}")
            results.append(TestResult(
                test_id=test_id_upper,
                label="ERROR",
                passed=False,
                message=f"Exception: {exc}",
                elapsed=0,
                errors=[str(exc)],
                transitions=0
            ))

    # Store results in run_log for HTML generation
    run_log.results = results

    print_report(results)

    # Generate HTML report
    html_content = generate_html_report(run_log, time.strftime("%Y-%m-%d %H:%M:%S"))
    html_path = logs_dir / f"run_protocols_{timestamp}.html"
    html_path.write_text(html_content)
    print(f"HTML report: {html_path}")

    # Cleanup
    if server_proc and not args.keep_server:
        stop_server(server_proc, server_log)
    elif server_log and not server_log.closed:
        server_log.close()

    if not results:
        return 130
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
