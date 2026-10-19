from unittest.mock import patch

import httpx
import pytest

from huddleroom.config import Settings
from huddleroom.models.agent import Agent
from huddleroom.services.orchestration_agent_definition_review import assess_agent_definition
from tests.live_acceptance.fixtures import (
    AGENT_EFFORT, AGENT_MODEL, CLI_RUNTIME, G2G4_GOAL_CONSTRAINTS, G2G4_GOAL_TEXT, G2G4_ROSTER, GOAL_TEXT,
    PROVIDER,
)
from tests.live_acceptance.live_env import server_env_for_cli
from tests.live_acceptance.run_dir import IsolationError, build_run_dir
from tests.live_acceptance.seed import create_goal, seed_minimal_roster
from tests.live_acceptance.server_supervisor import Supervisor, free_port

pytestmark = pytest.mark.live


@pytest.fixture(autouse=True)
def clean_test_database():
    """This module's stateful checks use their own Supervisor databases."""
    yield


def test_g2g4_writer_definition_keeps_implementation_and_summarization():
    writer = next(agent for agent in G2G4_ROSTER if agent["key"] == "writer")
    assessment = assess_agent_definition(
        Agent(
            name=writer["name"], role=writer["role"], description=writer["description"],
            system_prompt=writer["system_prompt"], capabilities=writer["capabilities"],
            adapter_type="cli", cli_runtime=writer["cli_runtime"], provider=writer["provider"],
            model=writer["model"], config={"reasoning_effort": writer["reasoning_effort"]}, is_active=True,
        ),
        ["implementation", "summarization"],
    )
    assert assessment["approved_for_work_functions"] == ["implementation", "summarization"]


def test_g2g4_goal_constraint_is_sent_with_goal_shape():
    with patch("tests.live_acceptance.seed._post", return_value={"goal": {"id": "goal-id"}}) as post:
        create_goal("http://unused", "project-id", G2G4_GOAL_TEXT, goal_type="roadmap", weight=None,
                    constraints=G2G4_GOAL_CONSTRAINTS)
    assert post.call_args.args[1] == "/api/v1/projects/project-id/orchestration/goals"
    assert post.call_args.args[2]["constraints"] == G2G4_GOAL_CONSTRAINTS


def test_seed_roster_discovery(tmp_path):
    rd = build_run_dir(tmp_path, "seed")
    with Supervisor(rd, free_port()) as sup:
        ids = seed_minimal_roster(sup.base_url, rd)
        agents = httpx.get(f"{sup.base_url}/api/v1/agents").json()["items"]
        project = httpx.get(f"{sup.base_url}/api/v1/projects/{ids['project_id']}").json()
    assert len(agents) == 3 and len(ids) == 4
    assert sorted(a["role"] for a in agents) == ["developer", "manager", "reviewer"]
    by_name = {a["name"]: a for a in agents}
    blank = by_name["live-dev-one"]
    assert blank["description"] == "" and blank["system_prompt"] == ""
    others = [a for a in agents if a is not blank]
    assert all(a["description"].strip() and a["system_prompt"].strip() for a in others)
    assert project["workspace_path"] == str(rd.workspace.resolve())
    assert {(a["adapter_type"], a["cli_runtime"], a["provider"], a["model"]) for a in agents} == {
        ("cli", CLI_RUNTIME, PROVIDER, AGENT_MODEL)}
    assert all(a["config"].get("reasoning_effort") == AGENT_EFFORT == "medium" for a in agents)


def test_seed_roster_full(tmp_path):
    rd = build_run_dir(tmp_path, "seed")
    with Supervisor(rd, free_port()) as sup:
        ids = seed_minimal_roster(sup.base_url, rd, roster="full", blank_definition_key=None)
        agents = httpx.get(f"{sup.base_url}/api/v1/agents").json()["items"]
    assert len(agents) == 4 and len(ids) == 5
    assert sorted(a["role"] for a in agents) == ["developer", "developer", "manager", "reviewer"]
    assert all(a["system_prompt"].strip() and a["description"].strip() for a in agents)
    assert {(a["adapter_type"], a["cli_runtime"], a["provider"], a["model"]) for a in agents} == {
        ("cli", CLI_RUNTIME, PROVIDER, AGENT_MODEL)}
    assert all(a["config"].get("reasoning_effort") == AGENT_EFFORT == "medium" for a in agents)


def test_seed_roster_g2g4(tmp_path):
    rd = build_run_dir(tmp_path, "seed")
    with Supervisor(rd, free_port()) as sup:
        ids = seed_minimal_roster(sup.base_url, rd, roster="g2g4", blank_definition_key=None)
        agents = httpx.get(f"{sup.base_url}/api/v1/agents").json()["items"]
        goal = create_goal(sup.base_url, ids["project_id"], G2G4_GOAL_TEXT)
    by_name = {agent["name"]: agent for agent in agents}
    assert len(agents) == 4 and len(ids) == 5
    assert all(agent["description"].strip() and agent["system_prompt"].strip() for agent in agents)
    assert {
        name: (agent["cli_runtime"], agent["model"], agent["config"].get("reasoning_effort"))
        for name, agent in by_name.items()
    } == {
        "live-manager": ("codex", "gpt-6.1-sol", "low"),
        "live-writer": ("claude_code", "claude-haiku-5-5", "medium"),
        "live-analyst": ("claude_code", "claude-haiku-5-5", "medium"),
        "live-reviewer": ("claude_code", "claude-haiku-5-5", "medium"),
    }
    assert goal["goal"]["weight"] == "substantial"


def test_isolation_refused(tmp_path):
    rd = build_run_dir(tmp_path, "seed")
    object.__setattr__(rd, "db", build_run_dir(tmp_path, "other").db)
    with pytest.raises(IsolationError):
        seed_minimal_roster("http://127.0.0.1:9", rd)


def test_create_goal(tmp_path):
    rd = build_run_dir(tmp_path, "goal")
    with Supervisor(rd, free_port()) as sup:
        ids = seed_minimal_roster(sup.base_url, rd)
        goal = create_goal(sup.base_url, ids["project_id"], GOAL_TEXT)
        with pytest.raises(ValueError):
            create_goal(sup.base_url, ids["project_id"], GOAL_TEXT, goal_type="auto")
        detail = httpx.get(
            f"{sup.base_url}/api/v1/projects/{ids['project_id']}/orchestration/goals/{goal['goal']['id']}").json()
    assert goal["goal"]["id"] and goal["goal"]["goal_type"] == "outcome"
    assert goal["goal"]["weight"] == "substantial" == detail["goal"]["weight"]
    assert goal["goal"]["weight_overridden_by"] and detail["goal"]["weight_overridden_by"]


def test_server_env_for_cli(monkeypatch):
    env = server_env_for_cli("m-1", "low")
    assert env == {
        "HUDDLEROOM_ORCHESTRATION_BACKEND": "codex",
        "HUDDLEROOM_ORCHESTRATION_CLI_MODEL": "m-1",
        "HUDDLEROOM_ORCHESTRATION_EFFORT": "low",
    }
    assert not any(k.startswith("OPENAI_") or k == "ANTHROPIC_API_KEY" for k in env)
    assert server_env_for_cli()["HUDDLEROOM_ORCHESTRATION_EFFORT"] in ("low", "medium", "high")
    for k, v in server_env_for_cli().items():
        monkeypatch.setenv(k, v)
    cfg = Settings(_env_file=None)
    assert (cfg.orchestration_backend, cfg.orchestration_effort) == ("codex", "low")
    assert cfg.orchestration_cli_model
