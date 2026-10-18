"""Constants for live acceptance runs: the goal text and the small seeded roster."""

import os

GOAL_TEXT = "Add a CONTRIBUTING.md with three short sections to the repository"

# Defaults only; override per run with the environment variables below. Names are not validated against a product list.
# All agents run on the claude_code CLI (subscription login); bare model id, no provider prefix.
AGENT_MODEL = os.environ.get("LIVE_ACCEPTANCE_AGENT_MODEL", "claude-haiku-5-5")
# Passed as agent config "reasoning_effort" -> claude CLI --effort.
AGENT_EFFORT = os.environ.get("LIVE_ACCEPTANCE_AGENT_EFFORT", "medium")
CLI_RUNTIME = "claude_code"
PROVIDER = "anthropic"
ORCH_CLI_MODEL = os.environ.get("LIVE_ACCEPTANCE_ORCH_MODEL", "gpt-6.1-sol")
ORCH_EFFORT = os.environ.get("LIVE_ACCEPTANCE_ORCH_EFFORT", "low")

# Agent keys in the "discovery" roster (subset of ROSTER). Tests that need the full team use ROSTER.
DISCOVERY_ROSTER_KEYS = ("manager", "dev1", "reviewer")
DEFAULT_GOAL_WEIGHT = "substantial"

# No QA and no Python-backend developer, on purpose.
ROSTER = (
    {"key": "manager", "name": "live-manager", "role": "manager",
     "description": "Coordinates the team and decides who does what.",
     "system_prompt": "You are the team manager. Break requests into small tasks and assign them to the right person.",
     "capabilities": ["planning", "coordination"]},
    {"key": "dev1", "name": "live-dev-one", "role": "developer",
     "description": "Writes documentation and small code changes.",
     "system_prompt": "You are a developer. Make small, correct changes in the repository and report what you did.",
     "capabilities": ["implementation", "documentation"]},
    {"key": "dev2", "name": "live-dev-two", "role": "developer",
     "description": "Writes documentation and small code changes.",
     "system_prompt": "You are a developer. Make small, correct changes in the repository and report what you did.",
     "capabilities": ["implementation", "documentation"]},
    {"key": "reviewer", "name": "live-reviewer", "role": "reviewer",
     "description": "Reviews finished work against the request.",
     "system_prompt": "You are a reviewer. Check finished work against the request and say clearly what is wrong.",
     "capabilities": ["review"]},
)
