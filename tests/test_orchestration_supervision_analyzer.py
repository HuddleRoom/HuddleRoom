"""Supervision analyzer: preamble, progress contract, and continue requires wake_when."""
import json

import pytest

from huddleroom.config import settings
from huddleroom.services.orchestration_supervision_analyzer import (
    OrchestrationSupervisionAnalyzer,
    parse_supervision_assessment,
)


def _wake_when(**overrides):
    wake = {
        "events": [],
        "recheck_after_seconds": 300,
        "expected_result": "test wait",
    }
    wake.update(overrides)
    return wake


def _assessment(disposition):
    return {
        "changes": [], "risks": [], "useful_learning": [], "criterion_progress": [],
        "disposition": {
            "origin": "supervision", "reason": "why", "expected_result": "expected",
            "contract_version": "plan:1", **disposition,
        },
    }


def _system_text(request):
    return next(m["content"] for m in request["messages"] if m["role"] == "system")


def test_build_request_prepends_preamble_and_progress_contract():
    from huddleroom.services.orchestration_llm_decision_adapter import PROGRESS_CONTRACT

    request = OrchestrationSupervisionAnalyzer.build_request({"goal": {}}, project={"name": "Rally"})
    system = _system_text(request)

    assert system.startswith("You are the orchestration control plane")
    assert "Project: Rally" in system
    assert PROGRESS_CONTRACT in system
    assert system.index(PROGRESS_CONTRACT) < system.index("Treat supplied data as evidence")


def test_build_request_uses_goal_headline_from_payload():
    payload = {"goal": {"objective": "Ship the beta", "status": "active", "weight": "high", "secret": "x"}}

    system = _system_text(OrchestrationSupervisionAnalyzer.build_request(payload))

    assert "Goal: Ship the beta (weight: high, status: active)" in system


def test_build_request_documents_continue_wake_when():
    from huddleroom.services.orchestration_wake_when import wake_when_prompt_table

    system = _system_text(OrchestrationSupervisionAnalyzer.build_request({}))

    assert "continue" in system
    assert "request.wake_when" in system
    assert '"recheck_after_seconds":<int>' in system
    assert "Allowed wake events and matcher keys: " + wake_when_prompt_table() in system
    assert (
        "In supervision, a need for a human decision maps to the attention disposition and waiting maps to "
        "continue (with wake_when)" in system
    )


def test_parse_continue_without_wake_when_raises():
    for request in (None, {}, {"wake_when": None}):
        disposition = {"action_type": "continue"}
        if request is not None:
            disposition["request"] = request
        with pytest.raises(ValueError, match="wake_when"):
            parse_supervision_assessment(_assessment(disposition))


def test_parse_continue_normalizes_and_clamps_wake_when():
    raw_wake = _wake_when(recheck_after_seconds=10**9)
    payload = _assessment({"action_type": "continue", "request": {"wake_when": raw_wake}})
    snapshot = json.dumps(payload, sort_keys=True)

    parsed = parse_supervision_assessment(payload)

    wake = parsed.disposition["request"]["wake_when"]
    assert wake["recheck_after_seconds"] == settings.orchestration_wake_max_seconds
    assert wake["expected_result"] == "test wait"
    assert wake["events"] == []
    assert json.dumps(payload, sort_keys=True) == snapshot  # input not mutated


def test_parse_non_continue_does_not_require_wake_when():
    parsed = parse_supervision_assessment(_assessment({"action_type": "pause"}))

    assert parsed.disposition["action_type"] == "pause"
    assert "request" not in parsed.disposition


@pytest.mark.asyncio
async def test_assess_repairs_continue_without_wake_when():
    calls = []
    responses = [
        json.dumps(_assessment({"action_type": "continue"})),
        json.dumps(_assessment({"action_type": "continue", "request": {"wake_when": _wake_when()}})),
    ]

    async def completion_fn(**request):
        calls.append(request)
        return {"choices": [{"message": {"content": responses[len(calls) - 1]}}]}

    analyzer = OrchestrationSupervisionAnalyzer(completion_fn=completion_fn)
    result = await analyzer.assess({"goal": {"objective": "Ship"}})

    assert len(calls) == 2
    repair_messages = calls[1]["messages"]
    assert any("wake_when" in str(m["content"]) for m in repair_messages if m["role"] == "user")
    assert result.disposition["request"]["wake_when"]["expected_result"] == "test wait"
