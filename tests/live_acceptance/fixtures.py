"""Constants for live acceptance runs: the goal text and the small seeded roster."""

import os

GOAL_TEXT = "Add a CONTRIBUTING.md with three short sections to the repository"
G2G4_GOAL_TEXT = (
    "Add a one-page pricing section to README.md for a fictional demo product with Free, Pro, and Team tiers. "
    "Research and compare those tiers, write the Markdown, and have it reviewed. Also save the draft plan to PLAN.md."
)
G2G4_GOAL_CONSTRAINTS = {
    "roster_policy": "Use only the supplied active roster. Do not propose or create agents; record uncovered responsibilities as gaps with no owner."
}
# G2G4 text plus one audience/tone paragraph (no new work or verbs): a long, multi-paragraph objective.
G5G6_GOAL_TEXT = G2G4_GOAL_TEXT + "\n\n" + (
    "Background: the readers are small open-source teams deciding whether to try the demo product. "
    "They skim on phones between meetings, so the tone should stay plain, friendly and brief, with no "
    "marketing hype, no invented customer quotes and no claims about real competitors. Prices are "
    "illustrative placeholders only and carry no promise of future pricing or availability for anyone."
)

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


G2G4_ROSTER = (
    {"key": "manager", "name": "live-manager", "role": "manager",
     "description": "Coordinates the team and writes plans. Does not edit files.",
     "system_prompt": "You are the team manager. Coordinate the team and write plans. Do not edit files.",
     "capabilities": ["planning", "management"], "cli_runtime": "codex", "provider": "openai",
     "model": "gpt-6.1-sol", "reasoning_effort": "low"},
    {"key": "writer", "name": "live-writer", "role": "writer",
     "description": "Makes small Markdown/code edits and summarizes its implementation. Does not test, validate or review work.",
     "system_prompt": "Implement small Markdown or code edits and summarize your implementation. Do not independently test, validate, or review work.",
     "capabilities": ["implementation", "summarization"], "cli_runtime": "claude_code", "provider": "anthropic",
     "model": "claude-haiku-5-5", "reasoning_effort": "medium"},
    {"key": "analyst", "name": "live-analyst", "role": "analyst",
     "description": "Researches and compares options such as pricing tiers and summarizes research. Does not write repository files.",
     "system_prompt": "Research and compare options, then summarize your research. Do not write repository files.",
     "capabilities": ["investigation", "summarization"], "cli_runtime": "claude_code", "provider": "anthropic",
     "model": "claude-haiku-5-5", "reasoning_effort": "medium"},
    {"key": "reviewer", "name": "live-reviewer", "role": "reviewer",
     "description": "Reviews finished work against the request.",
     "system_prompt": "Review finished work against the request.",
     "capabilities": ["review"], "cli_runtime": "claude_code", "provider": "anthropic",
     "model": "claude-haiku-5-5", "reasoning_effort": "medium"},
)

# G2G4 without any manager candidate: live-manager is dropped, so manager selection has no agent with
# management fit and recommends human_as_manager. live-analyst takes "requirements" (a planning-profile
# capability that the management profile does not score), so planning still routes to it.
# live-analyst also copies the manager's runtime fields because it owns planning here, and the live
# planner checks expect a codex planner.
_G2G4_MANAGER_RUNTIME = {k: v for k, v in G2G4_ROSTER[0].items()
                         if k in ("cli_runtime", "provider", "model", "reasoning_effort")}
G2G4_HUMAN_ROSTER = tuple(
    {**spec, "capabilities": [*spec["capabilities"], "requirements"], **_G2G4_MANAGER_RUNTIME}
    if spec["key"] == "analyst" else spec
    for spec in G2G4_ROSTER if spec["key"] != "manager"
)
