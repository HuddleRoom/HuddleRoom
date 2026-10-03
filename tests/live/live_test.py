#!/usr/bin/env python3
"""HuddleRoom live integration test suite.

Usage:
    python tests/live/live_test.py [options]

Options:
    --port PORT           Port to use (default: 8001)
    --no-start-server     Don't start server; assume already running
    --keep-server         Don't stop the server after tests
    --tests T6,T7,...     Comma-separated list of test IDs to run (default: all)
    --verbose             Print full WS event payloads
    --timeout SECONDS     Per-meeting WS timeout (default: 120)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import httpx

try:
    import websockets
except ImportError:
    print("ERROR: websockets not installed. Run: pip install websockets")
    sys.exit(1)

# ── Constants ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECT_NAME = "huddleroom-live-test"
TERMINAL_EVENT_TYPES = {"meeting.concluded", "meeting.cancelled", "meeting.timeout"}

# ── Result tracking ────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    test_id: str
    label: str
    passed: bool
    message: str
    turns: int = 0
    decisions: int = 0
    elapsed: float = 0.0
    errors: list[str] = field(default_factory=list)
    meeting_type: str = "decision"
    turn_strategy: str = "round_robin"


@dataclass
class LogEntry:
    """Structured log entry for HTML report."""
    timestamp: str        # ISO format
    kind: str             # "http_get" | "http_post" | "ws_event" | "info"
    test_id: str | None   # which test this belongs to, or None for setup
    data: dict            # flexible payload


class RunLog:
    """Collects structured events during the run."""
    def __init__(self) -> None:
        self.entries: list[LogEntry] = []
        self.results: list[TestResult] = []
        self.test_conversations: dict[str, dict] = {}

    def add(self, kind: str, test_id: str | None, data: dict) -> None:
        """Add a structured log entry."""
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        self.entries.append(LogEntry(
            timestamp=timestamp,
            kind=kind,
            test_id=test_id,
            data=data
        ))


# ── Agent definitions ──────────────────────────────────────────────────────────

AGENTS = [
    {
        "name": "live-architect",
        "role": "architect",
        "provider": "openi",
        "model": "openai/gpt-5-nano",
        "system_prompt": (
            "You are a software architect. You champion clean abstractions, scalability, "
            "and maintainability. You push back firmly on shortcuts and over-simple solutions. "
            "In meetings, make your position clear and respond directly to others' arguments. "
            "Be concise — 2-4 sentences per turn."
        ),
    },
    {
        "name": "live-pragmatist",
        "role": "engineer",
        "provider": "openi",
        "model": "openai/gpt-5-nano",
        "system_prompt": (
            "You are a pragmatic engineer. You value working code and fast delivery over "
            "elegant design. You push back on over-engineering. In meetings, disagree with "
            "the architect when you think they are over-complicating things. "
            "Be concise — 2-4 sentences per turn."
        ),
    },
    {
        "name": "live-security",
        "role": "security",
        "provider": "openi",
        "model": "openai/gpt-5-nano",
        "system_prompt": (
            "You are a security engineer. You identify vulnerabilities, auth gaps, and data risks. "
            "You object assertively to any decision that introduces security risk. "
            "In meetings, be direct: state the risk and what must change. "
            "Be concise — 2-4 sentences per turn."
        ),
    },
    {
        "name": "live-pm",
        "role": "pm",
        "provider": "openi",
        "model": "openai/gpt-5-nano",
        "system_prompt": (
            "You are a product manager. You focus on user impact and delivery timelines. "
            "You mediate between technical views and push discussions toward concrete decisions. "
            "In meetings, acknowledge all sides and propose actionable compromises. "
            "Be concise — 2-4 sentences per turn."
        ),
    },
]


# ── Meeting test definitions ───────────────────────────────────────────────────

MEETING_TESTS = [
    {
        "id": "T6",
        "label": "Sanity",
        "meeting_type": "decision",
        "turn_strategy": "round_robin",
        "agents": ["live-architect", "live-pragmatist"],
        "organizer": None,
        "agenda": [
            {
                "order": 1,
                "title": "TypeScript vs Python",
                "description": "Should this project use TypeScript or Python for the backend? Each participant state your recommendation in 2 sentences.",
                "max_rounds": 1,
            }
        ],
        "verify": {
            "min_turns": 2,
            "min_decisions": 0,
            "speakers": ["live-architect", "live-pragmatist"],
        },
    },
    {
        "id": "T7",
        "label": "Architecture Debate",
        "meeting_type": "decision",
        "turn_strategy": "round_robin",
        "deadlock_strategy": "majority_rules",
        "agents": ["live-architect", "live-pragmatist", "live-pm"],
        "organizer": None,
        "agenda": [
            {
                "order": 1,
                "title": "Database choice",
                "description": "PostgreSQL vs MongoDB for user data storage. Each make your case, then reach a recommendation.",
                "options": ["PostgreSQL", "MongoDB"],
                "max_rounds": 2,
            },
            {
                "order": 2,
                "title": "Caching strategy",
                "description": "Redis vs in-memory cache. State your recommendation and why.",
                "options": ["Redis", "in-memory cache"],
                "max_rounds": 2,
            },
        ],
        "verify": {
            "min_turns": 6,
            "min_decisions": 2,
            "speakers": ["live-architect", "live-pragmatist", "live-pm"],
            "require_non_empty_turns": True,
            "require_position_lines": True,
        },
    },
    {
        "id": "T8",
        "label": "Security Review",
        "meeting_type": "review",
        "turn_strategy": "moderated",
        "agents": ["live-architect", "live-security", "live-pm"],
        "organizer": None,
        "max_duration_minutes": 8,
        "agenda": [
            {
                "order": 1,
                "title": "Auth mechanism review",
                "description": "The API uses HTTP Basic Auth over HTTPS. Security: identify risks. Others: respond. Reach approve/reject.",
                "max_rounds": 2,
            },
            {
                "order": 2,
                "title": "Password storage review",
                "description": "User passwords are stored as MD5 hashes. Security: assess risk. Others: respond. Reach approve/reject.",
                "max_rounds": 2,
            },
        ],
        "verify": {
            "min_turns": 4,
            "min_decisions": 0,
            "speakers": ["live-security", "live-architect", "live-pm"],
            "require_non_empty_turns": True,
            "first_speaker": "live-security",
            "require_security_severity_lines": True,
            "require_all_agenda_items_completed": True,
            "required_resolution_kinds": [
                "approved",
                "approved_with_followups",
                "rejected",
                "deferred",
            ],
            "require_non_reviewer_response_per_item": True,
            "require_action_items_for_followups": True,
        },
    },
    {
        "id": "T9",
        "label": "Organizer Sprint Planning",
        "meeting_type": "decision",
        "turn_strategy": "organizer_controlled",
        "deadlock_strategy": "human_intervention",
        "max_duration_minutes": 30,
        "agents": ["live-architect", "live-pragmatist", "live-security", "live-pm"],
        "organizer": "live-pm",
        "agenda": [
            {
                "order": 1,
                "title": "Q3 top priority",
                "description": (
                    "The team must choose one Q3 anchor priority. Platform reliability means fixing outages "
                    "and reducing technical debt. Security hardening means closing auth and compliance gaps. "
                    "Feature delivery means shipping the two features sales promised this quarter. "
                    "Each participant: state your preferred option and two reasons. Do not defer to others — "
                    "argue your position based on your area of expertise."
                ),
                "question": "Which priority should anchor Q3 planning?",
                "options": [
                    "Platform reliability and technical debt reduction",
                    "Security hardening and compliance",
                    "New customer-facing feature delivery",
                ],
                "max_rounds": 2,
            },
            {
                "order": 2,
                "title": "API authentication strategy",
                "description": (
                    "Each participant is assigned a position to defend based on their role — this assignment is binding. "
                    "If you are an architect, you MUST argue for Token-based auth with short-lived JWTs. "
                    "If you are a security engineer, you MUST argue for Mutual TLS with certificate rotation. "
                    "If you are a pragmatist, you MUST argue for API keys with IP allowlisting. "
                    "If you are a product manager, you MUST argue for OAuth2 with service accounts. "
                    "You MUST NOT change your position regardless of what others say. "
                    "Give two reasons why your assigned option is the right choice."
                ),
                "question": "Which authentication strategy should be adopted for the new internal services API?",
                "options": [
                    "Token-based auth with short-lived JWTs",
                    "Mutual TLS with certificate rotation",
                    "API keys with IP allowlisting",
                    "OAuth2 with service accounts",
                ],
                "max_rounds": 1,
            },
            {
                "order": 3,
                "title": "Sprint velocity target",
                "description": (
                    "Set the sprint velocity target for Q3. Choose one."
                ),
                "question": "What sprint velocity target should be set for Q3?",
                "options": [
                    "Reduce velocity by 20% to allow quality focus",
                    "Maintain current velocity with quality gates",
                ],
                "max_rounds": 2,
            },
        ],
        "verify": {
            "min_turns": 6,
            "min_decisions": 1,
            "speakers": ["live-architect", "live-pragmatist", "live-security", "live-pm"],
            "require_all_agenda_items_completed": True,
            "required_resolution_kinds": ["consensus", "human_intervention", "majority"],
            "require_at_least_one_resolution_kind": "human_intervention",
            "require_terminal_event_type": "meeting.concluded",
            "require_final_status": "concluded",
            "forbid_agenda_statuses": ["abandoned"],
            "forbid_partial_meeting": True,
        },
    },
    {
        "id": "T10",
        "label": "Standup",
        "meeting_type": "standup",
        "turn_strategy": "round_robin",
        "agents": ["live-architect", "live-pragmatist", "live-security", "live-pm"],
        "organizer": None,
        "agenda": [
            {
                "order": 1,
                "title": "Daily standup",
                "description": "Each participant: state your current focus and one blocker in 2 sentences.",
                "max_rounds": 1,
            }
        ],
        "verify": {
            "min_turns": 4,
            "max_turns": 4,
            "min_decisions": 0,
            "max_decisions": 0,
            "speakers": ["live-architect", "live-pragmatist", "live-security", "live-pm"],
            "require_non_empty_turns": True,
            "require_standup_update_lines": True,
            "require_all_agenda_items_completed": True,
            "required_resolution_kinds": ["updates_shared"],
            "require_terminal_event_type": "meeting.concluded",
            "require_final_status": "concluded",
            "forbid_partial_meeting": True,
        },
    },
    {
        "id": "T11",
        "label": "Architecture Design Follow-up",
        "meeting_type": "decision",
        "turn_strategy": "moderated",
        "agents": ["live-architect", "live-pragmatist", "live-pm"],
        "organizer": None,
        "signal_check_enabled": True,
        "max_duration_minutes": 12,
        "agenda": [
            {
                "order": 1,
                "title": "Event bus architecture",
                "description": (
                    "Design the next iteration of the internal workflow platform. Debate a synchronous service mesh "
                    "versus an event-driven architecture. Reach a decision and name any concrete follow-up work "
                    "that should be tracked as action items."
                ),
                "question": "Should the workflow platform remain synchronous or adopt an event-driven architecture?",
                "options": ["Synchronous service mesh", "Event-driven architecture"],
                "max_rounds": 2,
            },
            {
                "order": 2,
                "title": "Migration plan",
                "description": (
                    "Agree on the first implementation step for the chosen architecture. If the team identifies "
                    "clear owners or deliverables, state them explicitly as action items."
                ),
                "question": "What should be the first migration step after the architecture choice?",
                "options": ["Write an ADR and spike the event bus", "Keep the current architecture and optimize APIs"],
                "max_rounds": 2,
            },
        ],
        "verify": {
            "min_turns": 4,
            "min_decisions": 1,
            "speakers": ["live-architect", "live-pragmatist", "live-pm"],
            "require_non_empty_turns": True,
            "require_position_lines": True,
            "require_all_agenda_items_completed": True,
            "require_terminal_event_type": "meeting.concluded",
            "require_final_status": "concluded",
            "forbid_partial_meeting": True,
            "require_action_items_visible": True,
            "require_moderator_trace_visible": True,
            "require_control_model_visible": True,
            "require_signal_probe_events": True,
        },
    },
]


# ── API client ─────────────────────────────────────────────────────────────────

class HuddleRoomClient:
    def __init__(self, base_url: str, log: logging.Logger | None = None, run_log: RunLog | None = None, test_id: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self.base_url, timeout=60)
        self.log = log
        self.run_log = run_log
        self.test_id = test_id

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

    def _post(self, path: str, body: dict) -> Any:
        r = self._http.post(path, json=body)
        if self.log:
            self.log.info(f"POST {path} body={json.dumps(body)} → {r.status_code}")
        if not r.is_success:
            raise RuntimeError(f"POST {path} → {r.status_code}: {r.text[:300]}")
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
        return resp_json

    def get_or_create_project(self, name: str) -> dict:
        page = self._get("/api/v1/projects", limit=200)
        for p in page.get("items", []):
            if p["name"] == name:
                print(f"  Reusing project '{name}' ({p['id']})")
                return p
        print(f"  Creating project '{name}'...")
        return self._post("/api/v1/projects", {"name": name, "description": "HuddleRoom live integration tests"})

    def _put(self, path: str, body: dict) -> Any:
        r = self._http.put(path, json=body)
        if self.log:
            self.log.info(f"PUT {path} body={json.dumps(body)} → {r.status_code}")
        if not r.is_success:
            raise RuntimeError(f"PUT {path} → {r.status_code}: {r.text[:300]}")
        resp_json = r.json()
        if self.log:
            self.log.info(json.dumps(resp_json))
        return resp_json

    def get_or_create_agent(self, defn: dict) -> dict:
        page = self._get("/api/v1/agents", limit=500)
        for a in page.get("items", []):
            if a["name"] == defn["name"]:
                needs_update = (
                    a.get("role") != defn["role"]
                    or a.get("provider") != defn["provider"]
                    or a.get("model") != defn["model"]
                    or a.get("system_prompt") != defn["system_prompt"]
                )
                if needs_update:
                    print(f"  Updating agent '{defn['name']}' ({a['id']}) — definition changed")
                    return self._put(f"/api/v1/agents/{a['id']}", {
                        "name": defn["name"],
                        "role": defn["role"],
                        "provider": defn["provider"],
                        "model": defn["model"],
                        "system_prompt": defn["system_prompt"],
                        "adapter_type": "api",
                    })
                print(f"  Reusing agent '{defn['name']}' ({a['id']})")
                return a
        print(f"  Creating agent '{defn['name']}'...")
        return self._post("/api/v1/agents", {
            "name": defn["name"],
            "role": defn["role"],
            "provider": defn["provider"],
            "model": defn["model"],
            "system_prompt": defn["system_prompt"],
            "adapter_type": "api",
        })

    def create_meeting(
        self,
        project_id: str,
        title: str,
        meeting_type: str,
        turn_strategy: str,
        participant_agent_ids: list[str],
        agenda_items: list[dict],
        organizer_agent_id: str | None = None,
        max_duration_minutes: int = 10,
        veto_window_hours: int = 0,
        deadlock_strategy: str | None = None,
        signal_check_enabled: bool = False,
    ) -> dict:
        body: dict[str, Any] = {
            "title": title,
            "meeting_type": meeting_type,
            "turn_strategy": turn_strategy,
            "participant_agent_ids": participant_agent_ids,
            "agenda_items": agenda_items,
            "auto_start": True,
            "max_duration_minutes": max_duration_minutes,
            "veto_window_hours": veto_window_hours,
            "signal_check_enabled": signal_check_enabled,
        }
        if organizer_agent_id:
            body["organizer_agent_id"] = organizer_agent_id
        if deadlock_strategy:
            body["deadlock_strategy"] = deadlock_strategy
        return self._post(f"/api/v1/projects/{project_id}/meetings", body)

    def list_turns(self, meeting_id: str) -> list[dict]:
        return self._get(f"/api/v1/meetings/{meeting_id}/turns") or []

    def list_decisions(self, meeting_id: str) -> list[dict]:
        return self._get(f"/api/v1/meetings/{meeting_id}/decisions") or []

    def get_meeting(self, meeting_id: str) -> dict:
        return self._get(f"/api/v1/meetings/{meeting_id}")

    def list_action_items(self, meeting_id: str) -> list[dict]:
        return self._get(f"/api/v1/meetings/{meeting_id}/action-items") or []

    def list_knowledge(self, project_id: str) -> list[dict]:
        return (self._get(f"/api/v1/projects/{project_id}/knowledge") or {}).get("items", [])

    def delete_knowledge(self, item_id: str) -> None:
        r = self._http.delete(f"/api/v1/knowledge/{item_id}")
        if self.log:
            self.log.info(f"DELETE /api/v1/knowledge/{item_id} → {r.status_code}")


# Compatibility for existing live-test imports.
RallyClient = HuddleRoomClient


# ── Fixture setup ──────────────────────────────────────────────────────────────

def setup_fixtures(client: RallyClient, log: logging.Logger | None = None) -> tuple[dict, dict[str, dict]]:
    """Create or reuse project and all 4 agents. Returns (project, agents_by_name)."""
    print("\n=== Setting up fixtures ===")
    if log:
        log.info("=== Setting up fixtures ===")
    project = client.get_or_create_project(PROJECT_NAME)

    # Clear stale knowledge items that bias agent reasoning across test runs
    knowledge_items = client.list_knowledge(project["id"])
    if knowledge_items:
        print(f"  Clearing {len(knowledge_items)} knowledge items from previous runs...")
        for ki in knowledge_items:
            client.delete_knowledge(ki["id"])

    agents_by_name: dict[str, dict] = {}
    for defn in AGENTS:
        agent = client.get_or_create_agent(defn)
        agents_by_name[defn["name"]] = agent

    print(f"  Project: {project['id']}")
    for name, a in agents_by_name.items():
        print(f"  Agent {name}: {a['id']} ({a['model']})")
    return project, agents_by_name


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HuddleRoom live integration tests")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--no-start-server", action="store_true")
    p.add_argument("--keep-server", action="store_true")
    p.add_argument("--tests", default="T6,T7,T8,T9,T10")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--timeout", type=int, default=720)
    return p.parse_args()


# ── Server management ──────────────────────────────────────────────────────────

def check_server(base_url: str) -> bool:
    try:
        r = httpx.get(f"{base_url}/api/v1/config", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


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
    expected_database_url = f"sqlite+aiosqlite:///{project_root / 'tests' / 'live' / 'huddleroom_live_test.db'}"
    env = os.environ.copy()
    env["HUDDLEROOM_DATABASE_URL"] = expected_database_url
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
    log_path = logs_dir / f"run_{timestamp}.log"

    logger = logging.getLogger("huddleroom_live_test")
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    print(f"Log file: {log_path}")
    return logger, timestamp, logs_dir


# ── WebSocket monitor ──────────────────────────────────────────────────────────

async def watch_meeting_ws(
    meeting_id: str,
    ws_base: str,
    timeout: int,
    verbose: bool,
    log: logging.Logger | None = None,
    run_log: RunLog | None = None,
    test_id: str | None = None,
) -> list[dict]:
    """Connect to meeting WS, stream events to stdout, return all events received."""
    uri = f"{ws_base}/ws/meetings/{meeting_id}"
    events: list[dict] = []

    try:
        async with websockets.connect(uri, ping_interval=20, ping_timeout=30) as ws:
            async def _recv() -> None:
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    events.append(msg)
                    ev_type = msg.get("event_type", msg.get("type", "?"))
                    if log:
                        log.info(f"[WS] meeting_id={meeting_id} {json.dumps(msg)}")
                    if run_log:
                        run_log.add("ws_event", test_id, {
                            "meeting_id": meeting_id,
                            "event_type": ev_type,
                            "payload": msg
                        })
                    if verbose:
                        print(f"    [WS] {ev_type}: {json.dumps(msg)}")
                    else:
                        turn_info = ""
                        if "turn" in msg:
                            t = msg["turn"]
                            speaker = t.get("speaker_agent_id", "human")
                            turn_info = f" turn={t.get('turn_number')} speaker={str(speaker)[:8]}"
                        print(f"    [WS] {ev_type}{turn_info}")
                    if ev_type in TERMINAL_EVENT_TYPES:
                        break

            await asyncio.wait_for(_recv(), timeout=timeout)

    except asyncio.TimeoutError:
        print(f"    [WS] TIMEOUT after {timeout}s — meeting did not conclude")
    except Exception as exc:
        print(f"    [WS] ERROR: {exc}")

    return events


# ── Meeting test runner ────────────────────────────────────────────────────────

async def wait_for_meeting_stability(
    client: RallyClient,
    meeting_id: str,
    settle_seconds: int = 20,
    poll_interval: float = 2.0,
) -> tuple[list[dict], list[dict], str]:
    def _read_state() -> tuple[list[dict], list[dict], str]:
        turns = client.list_turns(meeting_id)
        decisions = client.list_decisions(meeting_id)
        status = client.get_meeting(meeting_id).get("status", "unknown")
        return turns, decisions, status

    deadline = time.monotonic() + settle_seconds
    last_signature: tuple[str, int, int] | None = None
    stable_reads = 0

    while time.monotonic() < deadline:
        turns, decisions, status = await asyncio.to_thread(_read_state)
        signature = (status, len(turns), len(decisions))

        if signature == last_signature:
            stable_reads += 1
            if stable_reads >= 2:
                return turns, decisions, status
        else:
            stable_reads = 0
            last_signature = signature

        await asyncio.sleep(poll_interval)

    return await asyncio.to_thread(_read_state)

def _is_non_empty_turn(turn: dict) -> bool:
    return bool((turn.get("content") or "").strip())


def _has_position_line(turn: dict) -> bool:
    content = turn.get("content") or ""
    marker = "POSITION:"
    if marker not in content:
        return False
    _, _, remainder = content.partition(marker)
    return bool(remainder.strip())


def _has_review_severity_line(turn: dict) -> bool:
    content = turn.get("content") or ""
    return bool(re.search(r"\[(severity:\s*(blocker|major|minor)|blocker|major|minor)\]", content, re.IGNORECASE))


def _has_standup_update_lines(turn: dict) -> bool:
    content = (turn.get("content") or "").strip()
    if not content:
        return False
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(lines) != 3:
        return False
    required_prefixes = ("DONE:", "NOW:", "BLOCKERS:")
    for prefix, line in zip(required_prefixes, lines):
        if not line.upper().startswith(prefix):
            return False
        _, _, remainder = line.partition(":")
        if not remainder.strip():
            return False
    return True


def _covers_topic_groups(turns: list[dict], topic_groups: list[list[str]]) -> list[list[str]]:
    conversation = "\n".join((turn.get("content") or "").lower() for turn in turns)
    missing_groups: list[list[str]] = []
    for group in topic_groups:
        if not any(marker.lower() in conversation for marker in group):
            missing_groups.append(group)
    return missing_groups


def _agenda_items_by_id(meeting: dict) -> dict[str, dict]:
    return {str(item["id"]): item for item in meeting.get("agenda_items", [])}


def _reviewer_turns_by_item(turns: list[dict], reviewer_agent_id: str) -> dict[str, list[dict]]:
    per_item: dict[str, list[dict]] = {}
    for turn in turns:
        if str(turn.get("speaker_agent_id")) != reviewer_agent_id:
            continue
        item_id = turn.get("agenda_item_id")
        if item_id:
            per_item.setdefault(str(item_id), []).append(turn)
    return per_item


def _has_non_reviewer_response_after_reviewer(turns: list[dict], reviewer_agent_id: str, item_id: str) -> bool:
    reviewer_spoke = False
    for turn in turns:
        if str(turn.get("agenda_item_id")) != item_id:
            continue
        speaker_id = str(turn.get("speaker_agent_id"))
        if speaker_id == reviewer_agent_id:
            reviewer_spoke = True
            continue
        if reviewer_spoke and _is_non_empty_turn(turn):
            return True
    return False


def collect_verification_errors(
    *,
    verify: dict,
    turns: list[dict],
    decisions: list[dict],
    events: list[dict],
    final_meeting: dict,
    final_status: str,
    action_items: list[dict],
    agent_id_to_name: dict[str, str],
    agents_by_name: dict[str, dict],
    concluded: bool,
) -> list[str]:
    errors: list[str] = []
    speakers_seen: set[str] = set()
    for turn in turns:
        agent_id = turn.get("speaker_agent_id")
        if agent_id and str(agent_id) in agent_id_to_name:
            speakers_seen.add(agent_id_to_name[str(agent_id)])

    if len(turns) < verify["min_turns"]:
        errors.append(f"Expected ≥{verify['min_turns']} turns, got {len(turns)}")
    if "max_turns" in verify and len(turns) > verify["max_turns"]:
        errors.append(f"Expected ≤{verify['max_turns']} turns, got {len(turns)}")
    if len(decisions) < verify["min_decisions"]:
        errors.append(f"Expected ≥{verify['min_decisions']} decisions, got {len(decisions)}")
    if "max_decisions" in verify and len(decisions) > verify["max_decisions"]:
        errors.append(f"Expected ≤{verify['max_decisions']} decisions, got {len(decisions)}")
    if verify.get("require_non_empty_turns"):
        empty_turn_numbers = [
            turn.get("turn_number")
            for turn in turns
            if not _is_non_empty_turn(turn)
        ]
        if empty_turn_numbers:
            errors.append(f"Empty turn content for turns: {empty_turn_numbers}")
    if verify.get("require_position_lines"):
        missing_position_turns = [
            turn.get("turn_number")
            for turn in turns
            if _is_non_empty_turn(turn) and not _has_position_line(turn)
        ]
        if missing_position_turns:
            errors.append(f"Missing POSITION line for turns: {missing_position_turns}")
    if verify.get("require_standup_update_lines"):
        malformed_turns = [
            turn.get("turn_number")
            for turn in turns
            if _is_non_empty_turn(turn) and not _has_standup_update_lines(turn)
        ]
        if malformed_turns:
            errors.append(f"Missing DONE/NOW/BLOCKERS lines for turns: {malformed_turns}")
    for expected_speaker in verify["speakers"]:
        if expected_speaker not in speakers_seen:
            errors.append(f"Expected agent '{expected_speaker}' to speak, but did not")
    if verify.get("first_speaker"):
        first_turn_agent_id = turns[0].get("speaker_agent_id") if turns else None
        actual_first_speaker = agent_id_to_name.get(str(first_turn_agent_id)) if first_turn_agent_id else None
        if actual_first_speaker != verify["first_speaker"]:
            errors.append(
                f"Expected first speaker '{verify['first_speaker']}', got '{actual_first_speaker or 'none'}'"
            )
    if verify.get("require_security_severity_lines"):
        reviewer_agent_id = str(agents_by_name["live-security"]["id"])
        missing_severity_turns = [
            turn.get("turn_number")
            for turn in turns
            if str(turn.get("speaker_agent_id")) == reviewer_agent_id
            and _is_non_empty_turn(turn)
            and not _has_review_severity_line(turn)
        ]
        if missing_severity_turns:
            errors.append(
                "Security review turns missing categorized severity findings: "
                f"{missing_severity_turns}"
            )
    if verify.get("require_all_agenda_items_completed"):
        agenda_items = final_meeting.get("agenda_items", [])
        incomplete = [
            {"title": item.get("title"), "status": item.get("status")}
            for item in agenda_items
            if item.get("status") in {"pending", "active"}
        ]
        if incomplete:
            errors.append(f"Agenda items not completed: {incomplete}")
        required_resolution_kinds = set(verify.get("required_resolution_kinds", []))
        if required_resolution_kinds:
            invalid_resolution_items = [
                {
                    "title": item.get("title"),
                    "resolution_kind": item.get("resolution_kind"),
                }
                for item in agenda_items
                if item.get("status") not in {"pending", "active"}
                and item.get("resolution_kind") not in required_resolution_kinds
            ]
            if invalid_resolution_items:
                errors.append(f"Agenda items missing required resolution kinds: {invalid_resolution_items}")
    if verify.get("require_at_least_one_resolution_kind"):
        required_kind = verify["require_at_least_one_resolution_kind"]
        found = any(
            item.get("resolution_kind") == required_kind
            for item in final_meeting.get("agenda_items", [])
        )
        if not found:
            errors.append(
                f"No agenda item resolved with resolution_kind='{required_kind}'; "
                f"found: {[item.get('resolution_kind') for item in final_meeting.get('agenda_items', [])]}"
            )
    if verify.get("forbid_agenda_statuses"):
        disallowed_statuses = set(verify["forbid_agenda_statuses"])
        forbidden_items = [
            {"title": item.get("title"), "status": item.get("status")}
            for item in final_meeting.get("agenda_items", [])
            if item.get("status") in disallowed_statuses
        ]
        if forbidden_items:
            errors.append(f"Agenda items reached forbidden statuses: {forbidden_items}")
    if verify.get("require_non_reviewer_response_per_item"):
        reviewer_agent_id = str(agents_by_name["live-security"]["id"])
        agenda_by_id = _agenda_items_by_id(final_meeting)
        reviewer_turns = _reviewer_turns_by_item(turns, reviewer_agent_id)
        missing_response_items = []
        for item_id in reviewer_turns:
            if not _has_non_reviewer_response_after_reviewer(turns, reviewer_agent_id, item_id):
                missing_response_items.append(agenda_by_id.get(item_id, {}).get("title", item_id))
        if missing_response_items:
            errors.append(f"No non-reviewer response after reviewer on items: {missing_response_items}")
    if verify.get("require_action_items_for_followups"):
        agenda_items = final_meeting.get("agenda_items", [])
        followup_needed = [
            item.get("title")
            for item in agenda_items
            if item.get("resolution_kind") in {"rejected", "approved_with_followups", "deferred"}
        ]
        if followup_needed and not action_items:
            errors.append(f"Expected follow-up action items for review outcomes on: {followup_needed}")
    if verify.get("require_action_items_visible") and not action_items:
        errors.append("Expected at least one visible action item, but none were recorded")
    if verify.get("require_moderator_trace_visible"):
        moderator_turns = [
            turn
            for turn in turns
            if (turn.get("organizer_selection") or {}).get("selector_type") == "moderator"
        ]
        if not moderator_turns:
            errors.append("Expected moderator speaker-selection metadata on at least one persisted turn")
    if verify.get("require_control_model_visible"):
        selector_turns = [
            turn
            for turn in turns
            if (turn.get("organizer_selection") or {}).get("model_used")
        ]
        if not selector_turns:
            errors.append("Expected persisted selector metadata with the control model used")
    if verify.get("require_signal_probe_events"):
        signal_probe_events = [
            event for event in events
            if event.get("event_type", event.get("type", "")) == "meeting.signal_probe"
        ]
        if not signal_probe_events:
            errors.append("Expected meeting.signal_probe events, but none were observed")
    if verify.get("require_terminal_event_type"):
        required_terminal = verify["require_terminal_event_type"]
        terminal_event_types = [
            event.get("event_type", event.get("type", ""))
            for event in events
            if event.get("event_type", event.get("type", "")) in TERMINAL_EVENT_TYPES
        ]
        if required_terminal not in terminal_event_types:
            errors.append(
                f"Expected terminal WS event '{required_terminal}', got {terminal_event_types or ['none']}"
            )
    if not concluded and final_status == "active":
        errors.append("Meeting did not conclude during WS watch and remained active after stabilization polling")
    if verify.get("require_final_status") and final_status != verify["require_final_status"]:
        errors.append(
            f"Final meeting status is '{final_status}', expected '{verify['require_final_status']}'"
        )
    if verify.get("forbid_partial_meeting") and final_meeting.get("is_partial"):
        errors.append("Meeting was marked partial, but a full conclusion is required")
    if final_status not in ("concluded", "cancelled"):
        errors.append(f"Final meeting status is '{final_status}', expected 'concluded' or 'cancelled'")
    return errors


async def run_meeting_test(
    test_def: dict,
    client: RallyClient,
    project: dict,
    agents_by_name: dict[str, dict],
    ws_base: str,
    timeout: int,
    verbose: bool,
    log: logging.Logger | None = None,
    run_log: RunLog | None = None,
) -> TestResult:
    test_id = test_def["id"]
    label = test_def["label"]
    print(f"\n{'='*60}")
    print(f"  {test_id}: {label}")
    print(f"  type={test_def['meeting_type']} strategy={test_def['turn_strategy']}")
    print(f"  agents={test_def['agents']}")
    print(f"{'='*60}")
    if log:
        log.info(f"\n{'='*60}")
        log.info(f"{test_id}: {label}")
        log.info(f"type={test_def['meeting_type']} strategy={test_def['turn_strategy']}")
        log.info(f"agents={test_def['agents']}")
        log.info(f"{'='*60}")
    t_start = time.monotonic()

    # Set test_id on client for structured logging
    if run_log:
        client.test_id = test_id
        client.run_log = run_log

    errors: list[str] = []

    # Resolve agent IDs
    participant_ids = []
    for name in test_def["agents"]:
        if name not in agents_by_name:
            errors.append(f"Agent '{name}' not found in fixtures")
        else:
            participant_ids.append(str(agents_by_name[name]["id"]))

    organizer_id: str | None = None
    if test_def.get("organizer"):
        org_name = test_def["organizer"]
        if org_name in agents_by_name:
            organizer_id = str(agents_by_name[org_name]["id"])
        else:
            errors.append(f"Organizer agent '{org_name}' not found")

    if errors:
        return TestResult(test_id=test_id, label=label, passed=False,
                          message="Fixture resolution failed", errors=errors)

    # Create meeting
    try:
        meeting = client.create_meeting(
            project_id=str(project["id"]),
            title=f"[Live Test] {test_id}: {label}",
            meeting_type=test_def["meeting_type"],
            turn_strategy=test_def["turn_strategy"],
            participant_agent_ids=participant_ids,
            agenda_items=test_def["agenda"],
            organizer_agent_id=organizer_id,
            max_duration_minutes=test_def.get("max_duration_minutes", 10),
            deadlock_strategy=test_def.get("deadlock_strategy"),
            signal_check_enabled=test_def.get("signal_check_enabled", False),
        )
        meeting_id = meeting["id"]
        print(f"  Meeting created: {meeting_id}")
        print(f"  Status: {meeting['status']}")
    except Exception as exc:
        return TestResult(test_id=test_id, label=label, passed=False,
                          message=f"Meeting creation failed: {exc}", errors=[str(exc)])

    # Small delay so server starts the meeting async tasks
    await asyncio.sleep(1)

    # Watch WS
    print(f"  Watching WS (timeout={timeout}s)...")
    events = await watch_meeting_ws(meeting_id, ws_base, timeout, verbose, log=log, run_log=run_log, test_id=test_id)

    # Verify via REST
    concluded = any(
        e.get("event_type", e.get("type", "")) in TERMINAL_EVENT_TYPES
        for e in events
    )

    try:
        if concluded:
            turns = client.list_turns(meeting_id)
            decisions = client.list_decisions(meeting_id)
            final_meeting = client.get_meeting(meeting_id)
            action_items = client.list_action_items(meeting_id)
            final_status = final_meeting.get("status", "unknown")
        else:
            print("  WS ended without terminal event; polling for final persisted state...")
            turns, decisions, final_status = await wait_for_meeting_stability(client, meeting_id)
            final_meeting = client.get_meeting(meeting_id)
            action_items = client.list_action_items(meeting_id)
    except Exception as exc:
        turns = []
        decisions = []
        action_items = []
        final_meeting = {}
        final_status = "unknown"
        errors.append(f"Could not fetch stabilized meeting state: {exc}")

    elapsed = time.monotonic() - t_start

    transport_outcome = "terminal WS event received" if concluded else "no terminal WS event received"
    print(f"  Transport outcome: {transport_outcome}")
    print(f"  Final status: {final_status}")
    print(f"  Turns: {len(turns)}, Decisions: {len(decisions)}")

    agent_id_to_name = {str(v["id"]): k for k, v in agents_by_name.items()}

    # Store conversation data in run_log
    if run_log:
        run_log.test_conversations[test_id] = {
            "turns": turns,
            "decisions": decisions,
            "action_items": action_items,
            "agenda_items": final_meeting.get("agenda_items", []),
            "meeting": final_meeting,
            "agent_id_to_name": agent_id_to_name,
        }

    # Verify
    verify = test_def["verify"]
    errors.extend(
        collect_verification_errors(
            verify=verify,
            turns=turns,
            decisions=decisions,
            events=events,
            final_meeting=final_meeting,
            final_status=final_status,
            action_items=action_items,
            agent_id_to_name=agent_id_to_name,
            agents_by_name=agents_by_name,
            concluded=concluded,
        )
    )

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
        turns=len(turns),
        decisions=len(decisions),
        elapsed=elapsed,
        errors=errors,
        meeting_type=test_def.get("meeting_type", "decision"),
        turn_strategy=test_def.get("turn_strategy", "round_robin"),
    )


# ── Report ─────────────────────────────────────────────────────────────────────

def print_report(results: list[TestResult]) -> None:
    print(f"\n{'='*60}")
    print("  HUDDLEROOM LIVE TEST REPORT")
    print(f"{'='*60}")
    print(f"  {'ID':<6} {'Label':<25} {'Result':<8} {'Turns':<7} {'Dec':<5} {'Time':>7}")
    print(f"  {'-'*6} {'-'*25} {'-'*8} {'-'*7} {'-'*5} {'-'*7}")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  {r.test_id:<6} {r.label:<25} {status:<8} {r.turns:<7} {r.decisions:<5} {r.elapsed:>5.1f}s")
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


# ── Async test loop ────────────────────────────────────────────────────────────

async def run_tests(
    test_ids: list[str],
    client: RallyClient,
    project: dict,
    agents_by_name: dict[str, dict],
    ws_base: str,
    timeout: int,
    verbose: bool,
    log: logging.Logger | None = None,
    run_log: RunLog | None = None,
) -> list[TestResult]:
    selected = [t for t in MEETING_TESTS if t["id"] in test_ids]
    if not selected:
        print(f"No matching tests for IDs: {test_ids}")
        return []

    results: list[TestResult] = []
    for test_def in selected:
        result = await run_meeting_test(
            test_def=test_def,
            client=client,
            project=project,
            agents_by_name=agents_by_name,
            ws_base=ws_base,
            timeout=timeout,
            verbose=verbose,
            log=log,
            run_log=run_log,
        )
        results.append(result)

    return results


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

        # Build conversation HTML if available
        conversation_html = ""
        if test_id in run_log.test_conversations:
            conv_data = run_log.test_conversations[test_id]
            turns = conv_data.get("turns", [])
            decisions = conv_data.get("decisions", [])
            action_items = conv_data.get("action_items", [])
            agenda_items = conv_data.get("agenda_items", [])
            agent_id_to_name = conv_data.get("agent_id_to_name", {})

            # Build lookup dicts
            decisions_by_item_id: dict[str, list[dict]] = {}
            for d in decisions:
                iid = str(d.get("agenda_item_id", ""))
                decisions_by_item_id.setdefault(iid, []).append(d)

            agenda_by_id: dict[str, dict] = {str(a["id"]): a for a in agenda_items}

            if turns:
                conversation_html = '<div class="conversation">'
                conversation_html += '<h4 style="margin-top: 0; margin-bottom: 1rem; color: #333;">Conversation</h4>'

                # Speaker color cycling
                speaker_colors: dict[str, int] = {}
                color_idx = 0
                last_item_id: str | None = None

                for i, turn in enumerate(turns):
                    agent_id = turn.get("speaker_agent_id")
                    agent_name = agent_id_to_name.get(str(agent_id) if agent_id else "", str(agent_id)[:8] if agent_id else "Unknown")
                    turn_num = turn.get("turn_number", "?")
                    content = turn.get("content", "")
                    model = turn.get("model_used", "?")
                    latency = turn.get("latency_ms", 0)
                    token_count = turn.get("token_count")
                    moderator_note = turn.get("moderator_note") or ""
                    prompt_messages = turn.get("prompt_messages") or []
                    raw_response = turn.get("raw_response") or ""
                    reasoning_content = turn.get("reasoning_content") or ""
                    organizer_selection = turn.get("organizer_selection") or {}
                    current_item_id = str(turn.get("agenda_item_id") or "")

                    # Assign color to speaker
                    if agent_name not in speaker_colors:
                        speaker_colors[agent_name] = color_idx % 6
                        color_idx += 1
                    speaker_class = speaker_colors[agent_name]

                    # Moderator note banner
                    moderator_html = ""
                    if moderator_note:
                        moderator_html = f'<div class="moderator-note">Moderator: {moderator_note}</div>'

                    # Prompt messages collapsible
                    prompt_html = ""
                    if prompt_messages:
                        msgs_html = ""
                        for msg in prompt_messages:
                            role = msg.get("role", "?")
                            msg_content = msg.get("content") or ""
                            msgs_html += f'<div class="prompt-message-role">{role}</div><div class="prompt-message-content">{msg_content}</div>'
                        prompt_html = f'<details class="turn-details"><summary>Prompt sent ({len(prompt_messages)} messages)</summary><div style="padding: 0.5rem 0;">{msgs_html}</div></details>'

                    # Reasoning collapsible
                    reasoning_html = ""
                    if reasoning_content:
                        reasoning_html = f'<details class="turn-details reasoning-details"><summary>Reasoning ({len(reasoning_content)} chars)</summary><pre class="prompt-message-content">{reasoning_content}</pre></details>'

                    # Raw response collapsible (only if different from content)
                    raw_html = ""
                    if raw_response and raw_response.strip() != content.strip():
                        raw_html = f'<details class="turn-details"><summary>Raw LLM response</summary><pre class="prompt-message-content">{raw_response}</pre></details>'

                    # Organizer selection display
                    organizer_html = ""
                    if organizer_selection:
                        sel_msgs = organizer_selection.get("messages") or []
                        sel_raw = organizer_selection.get("raw_response") or ""
                        sel_reason = organizer_selection.get("reason") or ""
                        sel_id = organizer_selection.get("next_speaker_id") or ""
                        sel_reasoning = organizer_selection.get("reasoning_content") or ""
                        sel_model = organizer_selection.get("model_used") or ""
                        selector_type = organizer_selection.get("selector_type") or "organizer"
                        selected_by = organizer_selection.get("selected_by") or ""
                        sel_name = agent_id_to_name.get(str(sel_id), sel_id[:8] if sel_id else "?")
                        sel_msgs_html = ""
                        for msg in sel_msgs:
                            role = msg.get("role", "?")
                            msg_content = msg.get("content") or ""
                            sel_msgs_html += f'<div class="prompt-message-role">{role}</div><div class="prompt-message-content">{msg_content}</div>'
                        sel_reasoning_html = ""
                        if sel_reasoning:
                            sel_reasoning_html = (
                                f'<div class="prompt-message-role">reasoning</div>'
                                f'<div class="prompt-message-content">{sel_reasoning}</div>'
                            )
                        sel_model_label = f' <span style="font-size:0.75em;opacity:0.7">[{sel_model}]</span>' if sel_model else ""
                        selector_label = "Moderator selected" if selector_type == "moderator" else "Organizer selected"
                        selected_by_label = (
                            f' <span style="font-size:0.75em;opacity:0.7">[{selected_by}]</span>'
                            if selected_by else ""
                        )
                        organizer_html = (
                            f'<details class="turn-details organizer-details">'
                            f'<summary>{selector_label}: {sel_name} — {sel_reason}{selected_by_label}{sel_model_label}</summary>'
                            f'<div style="padding: 0.5rem 0;">'
                            f'<div style="font-weight:bold;font-size:0.85em;margin-bottom:0.5rem;">Organizer prompt:</div>'
                            f'{sel_msgs_html}'
                            f'<div class="prompt-message-role">raw response</div>'
                            f'<div class="prompt-message-content">{sel_raw}</div>'
                            f'{sel_reasoning_html}'
                            f'</div></details>'
                        )

                    # Token info
                    token_info = f" | Tokens: {token_count}" if token_count else ""

                    conversation_html += f'''
            <div class="turn speaker-{speaker_class}">
                <div class="turn-speaker speaker-{speaker_class}">{agent_name} (Turn {turn_num})</div>
                {moderator_html}
                {organizer_html}
                <div class="turn-content">{content}</div>
                {prompt_html}
                {reasoning_html}
                {raw_html}
                <div class="turn-meta">
                    <span>Model: {model}</span>
                    <span>Latency: {latency}ms{token_info}</span>
                </div>
            </div>
            '''

                    # After last turn for an agenda item, emit decision + resolution summary
                    next_item_id = str(turns[i + 1].get("agenda_item_id") or "") if i + 1 < len(turns) else ""
                    if current_item_id and (next_item_id != current_item_id):
                        for dec in decisions_by_item_id.get(current_item_id, []):
                            title = dec.get("title", "Decision")
                            chosen = dec.get("chosen_option", "?")
                            rationale = dec.get("rationale", "")
                            decided_by = dec.get("decided_by", "?")
                            conversation_html += f'''
            <div class="decision-block">
                <div class="decision-block-title">{title}</div>
                <div class="decision-block-content"><strong>Chosen:</strong> {chosen}</div>
                <div class="decision-block-content"><strong>Decided by:</strong> {decided_by}</div>
                <div class="decision-block-content"><strong>Rationale:</strong> {rationale}</div>
            </div>
            '''
                        # Show resolution summary from agenda item
                        agenda_item = agenda_by_id.get(current_item_id, {})
                        resolution_summary = agenda_item.get("resolution_summary") or ""
                        resolution_kind = agenda_item.get("resolution_kind") or ""
                        if resolution_summary or resolution_kind:
                            kind_label = f"[{resolution_kind}] " if resolution_kind else ""
                            conversation_html += f'<div class="resolution-summary">{kind_label}{resolution_summary}</div>'

                conversation_html += '</div>'
                if action_items:
                    conversation_html += '<div class="action-items">'
                    conversation_html += '<h4 style="margin-top: 1rem; margin-bottom: 0.75rem; color: #333;">Action Items</h4>'
                    for action_item in action_items:
                        description = action_item.get("description", "")
                        status = action_item.get("status", "")
                        conversation_html += (
                            f'<div class="decision-block">'
                            f'<div class="decision-block-content"><strong>{description}</strong></div>'
                            f'<div class="decision-block-content"><strong>Status:</strong> {status}</div>'
                            f'</div>'
                        )
                    conversation_html += '</div>'

        signal_events = []
        for entry in test_entries:
            if entry.kind != "ws_event":
                continue
            payload = entry.data.get("payload", {})
            event_type = entry.data.get("event_type", "?")
            if event_type == "meeting.signal_probe":
                signal_events.append(payload.get("payload", payload))

        if signal_events:
            conversation_html += '<div class="signal-probing">'
            conversation_html += '<h4 style="margin-top: 1rem; margin-bottom: 0.75rem; color: #333;">Signal Probing</h4>'
            for event in signal_events:
                probed_agent_id = str(event.get("probed_agent_id") or "")
                probed_name = agent_id_to_name.get(probed_agent_id, probed_agent_id[:8] if probed_agent_id else "Unknown")
                response = event.get("probe_response") or "(no response)"
                signal_message = event.get("signal_message")
                signaled = event.get("signaled")
                error = event.get("error")
                status_text = "Signal emitted" if signaled else "No signal emitted"
                if error:
                    status_text = f"Probe failed: {error}"
                conversation_html += (
                    f'<div class="decision-block">'
                    f'<div class="decision-block-title">{probed_name}</div>'
                    f'<div class="decision-block-content"><strong>Probe response:</strong> {response}</div>'
                    f'<div class="decision-block-content"><strong>{status_text}</strong></div>'
                )
                if signal_message:
                    conversation_html += (
                        f'<div class="decision-block-content"><strong>Signal message:</strong> {signal_message}</div>'
                    )
                conversation_html += '</div>'
            conversation_html += '</div>'

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
            elif entry.kind == "ws_event":
                ev_type = entry.data.get("event_type", "?")
                label = f"WS {ev_type}"
                payload = json.dumps(entry.data.get("payload", {}), indent=2)
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
                <div><strong>Type:</strong> {result.meeting_type}</div>
                <div><strong>Strategy:</strong> {result.turn_strategy}</div>
                <div><strong>Turns:</strong> {result.turns} | <strong>Decisions:</strong> {result.decisions} | <strong>Time:</strong> {result.elapsed:.1f}s</div>
            </div>
            <div style="margin-top: 1rem;">
                {conversation_html if conversation_html else ''}
                <div style="margin-top: 1rem; border-top: 1px solid #eee; padding-top: 1rem;">
                    <h4 style="margin-top: 0;">HTTP/WS Events</h4>
                    {entry_html if entry_html else '<p style="color: #999;">No events recorded</p>'}
                </div>
            </div>
        </section>
        '''

    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>HuddleRoom Live Test Report</title>
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
        .conversation {{
            padding: 1rem;
            margin-bottom: 1.5rem;
            background-color: #fafafa;
            border-radius: 4px;
            border: 1px solid #e0e0e0;
        }}
        .turn {{
            margin-bottom: 1rem;
            padding: 1rem;
            border-left: 4px solid #ccc;
            background-color: white;
            border-radius: 2px;
        }}
        .turn.speaker-0 {{ border-left-color: #ffb3ba; }}
        .turn.speaker-1 {{ border-left-color: #bae1ff; }}
        .turn.speaker-2 {{ border-left-color: #ffffba; }}
        .turn.speaker-3 {{ border-left-color: #baffc9; }}
        .turn.speaker-4 {{ border-left-color: #ffc9ba; }}
        .turn.speaker-5 {{ border-left-color: #e0bbff; }}
        .turn-speaker {{
            font-weight: bold;
            margin-bottom: 0.5rem;
            font-size: 0.95em;
        }}
        .turn-speaker.speaker-0 {{ color: #d32f2f; }}
        .turn-speaker.speaker-1 {{ color: #1976d2; }}
        .turn-speaker.speaker-2 {{ color: #f57f17; }}
        .turn-speaker.speaker-3 {{ color: #388e3c; }}
        .turn-speaker.speaker-4 {{ color: #c2185b; }}
        .turn-speaker.speaker-5 {{ color: #7b1fa2; }}
        .turn-content {{
            margin-bottom: 0.5rem;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            white-space: pre-wrap;
            word-wrap: break-word;
            line-height: 1.5;
        }}
        .turn-meta {{
            font-size: 0.85em;
            color: #999;
            border-top: 1px solid #f0f0f0;
            padding-top: 0.5rem;
        }}
        .turn-meta span {{
            margin-right: 1rem;
            display: inline-block;
        }}
        .decision-block {{
            margin: 1rem 0;
            padding: 1rem;
            background-color: #fffde7;
            border-left: 4px solid #fbc02d;
            border-radius: 2px;
        }}
        .decision-block-title {{
            font-weight: bold;
            color: #f57f17;
            margin-bottom: 0.5rem;
        }}
        .decision-block-content {{
            font-size: 0.95em;
            margin-bottom: 0.5rem;
        }}
        .moderator-note {{
            background-color: #fff8e1;
            border-left: 3px solid #fbc02d;
            padding: 0.4rem 0.75rem;
            margin-bottom: 0.5rem;
            font-style: italic;
            font-size: 0.9em;
            color: #5d4037;
        }}
        .turn-details {{
            margin-top: 0.5rem;
            font-size: 0.85em;
        }}
        .turn-details summary {{
            cursor: pointer;
            color: #1565c0;
            font-size: 0.9em;
        }}
        .prompt-message-role {{
            font-weight: bold;
            font-size: 0.8em;
            text-transform: uppercase;
            color: #555;
            margin-top: 0.75rem;
            margin-bottom: 0.25rem;
        }}
        .prompt-message-content {{
            background-color: #f8f8f8;
            border: 1px solid #e0e0e0;
            border-radius: 2px;
            padding: 0.5rem;
            font-family: monospace;
            font-size: 0.8em;
            white-space: pre-wrap;
            word-break: break-word;
        }}
        .resolution-summary {{
            font-size: 0.9em;
            color: #33691e;
            margin: 0.5rem 0 1rem 0;
            padding: 0.4rem 0.75rem;
            background-color: #f1f8e9;
            border-left: 3px solid #7cb342;
            border-radius: 2px;
        }}
        .organizer-details summary {{
            color: #7b1fa2;
        }}
        .reasoning-details summary {{
            background: #e8f5e9;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>HuddleRoom Live Test Report</h1>
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
                    <th>Turns</th>
                    <th>Decisions</th>
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
                    <td>{result.turns}</td>
                    <td>{result.decisions}</td>
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
    ws_url = f"ws://localhost:{args.port}"
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
            server_log_path = logs_dir / f"server_{timestamp}.log"
            try:
                server_proc, server_log = start_server(base_url, args.port, PROJECT_ROOT, server_log_path)
            except Exception as exc:
                print(f"ERROR: {exc} (server log: {server_log_path})")
                return 1

    client = HuddleRoomClient(base_url, log=log, run_log=run_log)

    try:
        project, agents_by_name = setup_fixtures(client, log=log)
    except Exception as exc:
        print(f"ERROR: Fixture setup failed: {exc}")
        if server_proc and not args.keep_server:
            stop_server(server_proc, server_log)
        elif server_log and not server_log.closed:
            server_log.close()
        return 1

    print(f"\nRunning tests: {', '.join(test_ids)}")
    print(f"WS timeout per test: {args.timeout}s\n")

    try:
        results = asyncio.run(run_tests(
            test_ids=test_ids,
            client=client,
            project=project,
            agents_by_name=agents_by_name,
            ws_base=ws_url,
            timeout=args.timeout,
            verbose=args.verbose,
            log=log,
            run_log=run_log,
        ))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        results = []
    finally:
        if server_proc and not args.keep_server:
            stop_server(server_proc, server_log)
        elif server_log and not server_log.closed:
            server_log.close()

    # Store results in run_log for HTML generation
    run_log.results = results

    print_report(results)

    # Generate HTML report
    html_content = generate_html_report(run_log, time.strftime("%Y-%m-%d %H:%M:%S"))
    html_path = logs_dir / f"run_{timestamp}.html"
    html_path.write_text(html_content)
    print(f"HTML report: {html_path}")

    if not results:
        return 130  # interrupted or no tests ran
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
