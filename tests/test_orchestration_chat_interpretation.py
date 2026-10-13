"""Provider intent is bounded advisory data, never executable authority."""
import json
import uuid

import pytest


def _instruction(goal_id, **changes):
    return {"intent": "instruction", "goal_id": str(goal_id), "operation": "resume", "payload": {}, **changes}


@pytest.mark.parametrize("content", [
    "Why is Alpha paused?", "Could you explain how to resume Alpha?", "What would happen if I resume Alpha?",
    'The note says "Resume Alpha". What does it mean?', '"Resume Alpha"',
    "Suppose I resume Alpha", "If I resume Alpha, what changes?", "Would pausing Alpha help?",
    "Is Alpha paused", "Explain Alpha's status", "Alpha is paused",
])
def test_non_authorizing_input_cannot_be_promoted_by_provider(content):
    from huddleroom.services.orchestration_chat_interpretation import parse_chat_interpretation

    goal_id = uuid.uuid4()
    parsed = parse_chat_interpretation(json.dumps(_instruction(goal_id)), content, goal_id)
    assert parsed.operation is None
    assert parsed.intent != "instruction"


@pytest.mark.parametrize("operation", ["reset", "create_goal", "stop_continuous", "waive_gate", "run_tool"])
def test_malformed_or_unsupported_output_never_yields_operation(operation):
    from huddleroom.services.orchestration_chat_interpretation import parse_chat_interpretation

    goal_id = uuid.uuid4()
    parsed = parse_chat_interpretation(json.dumps(_instruction(goal_id, operation=operation)), "Resume Alpha", goal_id)
    assert parsed.operation is None


@pytest.mark.parametrize("raw", ["not json", "[]", "null", '{"intent":"instruction"}',
                                   '{"intent":"instruction","operation":"resume","payload":{"command":"run"}}'])
def test_invalid_interpretation_clarifies_without_effect(raw):
    from huddleroom.services.orchestration_chat_interpretation import parse_chat_interpretation

    parsed = parse_chat_interpretation(raw, "Resume Alpha", uuid.uuid4())
    assert parsed.operation is None


def test_clear_supported_instruction_retains_typed_data():
    from huddleroom.services.orchestration_chat_interpretation import parse_chat_interpretation

    goal_id = uuid.uuid4()
    parsed = parse_chat_interpretation(json.dumps(_instruction(goal_id)), "Resume Alpha", goal_id)
    assert parsed.intent == "instruction" and parsed.operation == "resume" and parsed.goal_id == goal_id


def test_assistant_suggestion_is_not_promoted_to_instruction():
    from huddleroom.services.orchestration_chat_interpretation import parse_chat_interpretation

    goal_id = uuid.uuid4()
    raw = {"intent": "question", "answer": "You could resume Alpha.", "goal_id": str(goal_id), "operation": "resume"}
    parsed = parse_chat_interpretation(json.dumps(raw), "What should I do next?", goal_id)
    assert parsed.operation is None and parsed.intent != "instruction"


def test_ambiguous_acceptance_requires_bound_action():
    from huddleroom.services.orchestration_chat_interpretation import parse_chat_interpretation

    goal_id = uuid.uuid4()
    parsed = parse_chat_interpretation(json.dumps(_instruction(goal_id)), "yes", goal_id)
    assert parsed.operation is None


def test_conflicting_provider_target_without_named_goal_clarifies():
    from huddleroom.services.orchestration_chat_interpretation import parse_chat_interpretation

    parsed = parse_chat_interpretation(json.dumps(_instruction(uuid.uuid4())), "Resume this goal", uuid.uuid4())
    assert parsed.operation is None


def test_steering_defaults_to_remaining_current_run():
    from huddleroom.services.orchestration_chat_interpretation import parse_chat_interpretation

    goal_id = uuid.uuid4()
    parsed = parse_chat_interpretation(json.dumps(_instruction(goal_id, operation="submit_steering",
        payload={"directive": "Prioritize verification"})), "Prioritize verification", goal_id)
    assert (parsed.payload.scope, parsed.payload.lifetime, parsed.payload.target_type) == (
        "run", "remaining_current_run", "goal")


def test_explicit_steering_instruction_cannot_authorize_different_lifecycle_operation():
    from huddleroom.services.orchestration_chat_interpretation import parse_chat_interpretation

    goal_id = uuid.uuid4()
    parsed = parse_chat_interpretation(json.dumps(_instruction(goal_id)), "Prioritize verification", goal_id)
    assert parsed.operation is None
