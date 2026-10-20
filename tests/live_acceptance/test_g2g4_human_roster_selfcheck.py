"""Zero-token check: the g2g4_human roster leaves manager selection with no strong candidate."""

import uuid

import pytest

from huddleroom.models.agent import Agent
from huddleroom.services.orchestration_manager_selection import (
    MANAGER_WORK_FUNCTION,
    ManagerSelectionProcess,
    _is_strong_candidate,
)
from tests.live_acceptance.fixtures import G2G4_HUMAN_ROSTER

pytestmark = pytest.mark.live


def _agents():
    return [
        Agent(
            id=uuid.uuid4(),
            name=s["name"],
            role=s["role"],
            description=s["description"],
            system_prompt=s["system_prompt"],
            capabilities=s["capabilities"],
            adapter_type="cli",
            cli_runtime=s["cli_runtime"],
            provider=s["provider"],
            model=s["model"],
            config={"reasoning_effort": s["reasoning_effort"]},
            is_active=True,
        )
        for s in G2G4_HUMAN_ROSTER
    ]


def test_g2g4_human_roster_has_no_strong_manager_candidate(capsys):
    process = ManagerSelectionProcess()
    agents = _agents()
    by_id = {a.id: a for a in agents}
    fits = process.roster_mapper._rank_agent_list(agents, {}, MANAGER_WORK_FUNCTION)
    assert fits
    for fit in fits:
        bonus, signals, definition_strong = process._inspect_candidate_definition(
            by_id[fit.agent_id]
        )
        print(
            f"{fit.name}: score={fit.score} bonus={bonus} strong_def={definition_strong} signals={signals}"
        )
        assert not _is_strong_candidate(fit.score, bonus, definition_strong), fit.name
    planning = process.roster_mapper._rank_agent_list(agents, {}, "planning")
    print("planning:", [(f.name, f.score) for f in planning])
    assert planning[0].name == "live-analyst" and not planning[0].weak
