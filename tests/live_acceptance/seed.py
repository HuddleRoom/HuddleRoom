"""Seed a minimal roster into a running live server over REST, and create goals."""

from __future__ import annotations

import httpx

from tests.live_acceptance.fixtures import (
    AGENT_EFFORT, AGENT_MODEL, CLI_RUNTIME, DEFAULT_GOAL_WEIGHT, DISCOVERY_ROSTER_KEYS, PROVIDER, ROSTER,
)
from tests.live_acceptance.run_dir import RunDir, assert_isolated


def _post(base_url: str, path: str, body: dict) -> dict:
    r = httpx.post(f"{base_url}{path}", json=body, timeout=60)
    r.raise_for_status()
    return r.json()


def seed_minimal_roster(base_url: str, run_dir: RunDir, roster: str = "discovery",
                        blank_definition_key: str | None = "dev1") -> dict:
    """Create the project and CLI agents. roster 'discovery' = 3 agents, 'full' = all 4 ROSTER agents.
    The agent named by blank_definition_key gets empty description and system_prompt (None keeps all full).
    The server must be running on run_dir with auth disabled."""
    if roster not in ("discovery", "full"):
        raise ValueError(f"unknown roster {roster!r}")
    specs = [s for s in ROSTER if roster == "full" or s["key"] in DISCOVERY_ROSTER_KEYS]
    if blank_definition_key is not None and blank_definition_key not in {s["key"] for s in specs}:
        raise ValueError(f"blank_definition_key {blank_definition_key!r} is not in roster {roster!r}")
    assert_isolated(run_dir, run_dir.db, run_dir.workspace)
    assert run_dir.workspace.is_dir(), run_dir.workspace
    if httpx.get(f"{base_url}/api/v1/projects", timeout=10).status_code != 200:
        raise RuntimeError("live server must run with auth disabled")
    project = _post(base_url, "/api/v1/projects", {
        "name": "live-acceptance-project", "description": "Throwaway live acceptance project",
        "workspace_path": str(run_dir.workspace.resolve())})
    ids = {"project_id": project["id"]}
    for spec in specs:
        body = {k: spec[k] for k in ("name", "role", "description", "system_prompt", "capabilities")}
        if spec["key"] == blank_definition_key:
            body["description"] = ""
            body["system_prompt"] = ""
        body |= {"adapter_type": "cli", "cli_runtime": CLI_RUNTIME, "provider": PROVIDER, "model": AGENT_MODEL,
                    "config": {"reasoning_effort": AGENT_EFFORT}}
        ids[f"{spec['key']}_id"] = _post(base_url, "/api/v1/agents", body)["id"]
    return ids


def set_goal_weight(base_url: str, project_id: str, goal_id: str, weight: str = DEFAULT_GOAL_WEIGHT,
                    reason: str = "live acceptance fixture") -> dict:
    """POST a weight override (operator path, no model call). Raises on non-2xx."""
    return _post(base_url, f"/api/v1/projects/{project_id}/orchestration/goals/{goal_id}/weight",
                 {"weight": weight, "reason": reason})


def create_goal(base_url: str, project_id: str, text: str, goal_type: str = "outcome",
                allow_classifier: bool = False, weight: str | None = DEFAULT_GOAL_WEIGHT, **shape) -> dict:
    """POST a goal, then override its weight unless weight is None.
    goal_type 'auto' runs the goal-shape classifier (a model call) and needs allow_classifier=True."""
    if goal_type == "auto" and not allow_classifier:
        raise ValueError("goal_type='auto' makes a model call; pass allow_classifier=True to opt in")
    created = _post(base_url, f"/api/v1/projects/{project_id}/orchestration/goals",
                    {"objective": text, "goal_type": goal_type, **shape})
    if weight is None:
        return created
    return set_goal_weight(base_url, project_id, created["goal"]["id"], weight=weight)
