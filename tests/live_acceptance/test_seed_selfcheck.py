import httpx
import pytest

from huddleroom.config import Settings
from tests.live_acceptance.fixtures import AGENT_EFFORT, AGENT_MODEL, CLI_RUNTIME, GOAL_TEXT, PROVIDER
from tests.live_acceptance.live_env import server_env_for_cli
from tests.live_acceptance.run_dir import IsolationError, build_run_dir
from tests.live_acceptance.seed import create_goal, seed_minimal_roster
from tests.live_acceptance.server_supervisor import Supervisor, free_port

pytestmark = pytest.mark.live


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
