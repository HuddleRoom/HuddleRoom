"""Supervision analyzer: preamble, progress contract, and continue requires wake_when."""
import json

import pytest

from huddleroom.config import settings
from huddleroom.services.orchestration_supervision import DISPOSITIONS, disposition_request_fields
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
    assert "A need for an owner decision maps to ask_human, with the exact question." in system
    assert "`continue` immediately creates a bounded wait and does not itself cause another action." in system
    assert "`ask_human` creates a real owner decision with your exact question; `attention` only records a warning." in system
    assert "supervision must not set applies_decision_id" in system
    assert "set top-level applies_decision_id" not in system
    assert system.count("applies_decision_id") == 1


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


UUID_A = "11111111-1111-1111-1111-111111111111"
UUID_B = "22222222-2222-2222-2222-222222222222"
VALID_FIELDS = {
    "wake_when": _wake_when(), "agent_id": UUID_A, "task_id": UUID_A, "gate_id": UUID_A,
    "participant_agent_ids": [UUID_A, UUID_B], "deliverable": "report", "scope": "scope",
    "work_function": "review", "topic": "sync", "graph_id": UUID_A, "parent_task_id": UUID_A, "subject_id": UUID_B,
    "subject_type": "task", "question": "Which option?",
}


def _valid_request(kind, **overrides):
    required, _ = disposition_request_fields(kind)
    return {**{field: VALID_FIELDS[field] for field in required}, **overrides}


def test_build_request_lists_every_disposition_with_its_request_fields():
    system = _system_text(OrchestrationSupervisionAnalyzer.build_request({}))

    for kind in sorted(DISPOSITIONS):
        required, optional = disposition_request_fields(kind)
        line = next(line for line in system.splitlines() if line.startswith(f"- {kind}:"))
        assert all(field in line for field in required), kind
        assert all(field in line for field in optional), kind


def test_build_request_omits_non_dispatchable_names():
    system = _system_text(OrchestrationSupervisionAnalyzer.build_request({}))

    for name in ("request_human_decision", "request_manager_decision", "request_split"):
        assert name not in system


@pytest.mark.parametrize("kind", sorted(DISPOSITIONS))
def test_parse_rejects_each_missing_required_request_field(kind):
    required, _ = disposition_request_fields(kind)
    for field in required:
        request = _valid_request(kind)
        del request[field]
        with pytest.raises(ValueError, match=field):
            parse_supervision_assessment(_assessment({"action_type": kind, "request": request}))


@pytest.mark.parametrize("kind", sorted(DISPOSITIONS))
def test_parse_rejects_unknown_request_field(kind):
    with pytest.raises(ValueError, match="unknown request fields: bogus"):
        parse_supervision_assessment(_assessment({
            "action_type": kind, "request": {**_valid_request(kind), "bogus": 1},
        }))


@pytest.mark.parametrize("kind,field", [
    ("follow_up", "agent_id"), ("reassign", "task_id"), ("verify", "gate_id"),
    ("meeting", "participant_agent_ids"), ("ask_human", "gate_id"),
])
def test_parse_rejects_malformed_id_fields(kind, field):
    bad = "not-a-uuid" if field != "participant_agent_ids" else [UUID_A, "not-a-uuid"]
    with pytest.raises(ValueError, match=field):
        parse_supervision_assessment(_assessment({
            "action_type": kind, "request": _valid_request(kind, **{field: bad}),
        }))


def test_follow_up_requires_parent_task_id():
    request = _valid_request("follow_up")
    del request["parent_task_id"]
    with pytest.raises(ValueError, match="parent_task_id"):
        parse_supervision_assessment(_assessment({"action_type": "follow_up", "request": request}))


def test_parse_rejects_non_uuid_parent_task_id_and_bad_subject_type():
    with pytest.raises(ValueError, match="parent_task_id"):
        parse_supervision_assessment(_assessment({
            "action_type": "follow_up", "request": _valid_request("follow_up", parent_task_id="nope")}))
    with pytest.raises(ValueError, match="subject_type"):
        parse_supervision_assessment(_assessment({
            "action_type": "graph", "request": _valid_request("graph", subject_type="goal")}))


def test_parse_verify_without_work_function():
    parsed = parse_supervision_assessment(_assessment({"action_type": "verify", "request": {"gate_id": UUID_A}}))
    assert parsed.disposition["request"] == {"gate_id": UUID_A}


def test_parse_ask_human_disposition():
    parsed = parse_supervision_assessment(_assessment({
        "action_type": "ask_human", "request": {"question": "Which option?"},
    }))

    assert parsed.disposition["action_type"] == "ask_human"
    assert parsed.disposition["request"] == {"question": "Which option?"}


@pytest.mark.asyncio
async def test_assess_repairs_malformed_request_then_returns_valid_assessment():
    calls = []
    responses = [
        json.dumps(_assessment({"action_type": "follow_up", "request": {"agent_id": UUID_A}})),
        json.dumps(_assessment({"action_type": "follow_up", "request": _valid_request("follow_up")})),
    ]

    async def completion_fn(**request):
        calls.append(request)
        return {"choices": [{"message": {"content": responses[len(calls) - 1]}}]}

    analyzer = OrchestrationSupervisionAnalyzer(completion_fn=completion_fn)
    result = await analyzer.assess({"goal": {"objective": "Ship"}})

    assert len(calls) == 2
    assert result.disposition["action_type"] == "follow_up"
    assert result.disposition["request"]["deliverable"] == "report"
