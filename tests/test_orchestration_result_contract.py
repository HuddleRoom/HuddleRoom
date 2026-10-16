import pytest

from huddleroom.services.orchestration_result_contract import (
    expected_result_kind as kind,
    normalize_result_output as norm,
)


@pytest.mark.parametrize("contract,gate,expected", [
    (None, False, "work_report"),
    ({"expected_result": "canonical_work_report"}, False, "work_report"),
    ({"expected_result": "final_summary", "report_schema": {}}, True, "final_summary"),
    ({"expected_result": "bogus", "report_schema": {"verdict": 1}}, False, "verification_verdict"),
    ({"report_schema": {"plan_kind": "x"}}, False, "planning_envelope"),
    ({"report_schema": {"plan_kind": "x"}}, True, "work_report"),
    ({"report_schema": {"criteria": 1}}, False, "final_summary"),
    ({"report_schema": {"selected_option": 1}}, False, "decision_report"),
    ({"report_schema": {"candidates": 1}}, False, "continuous_candidates"),
    ({"report_schema": {"other": 1}}, False, "work_report"),
    ({"expected_result": ["x"], "report_schema": {"verdict": 1}}, False, "verification_verdict"),
    ({"expected_result": {"a": 1}}, False, "work_report"),
    (["not", "a", "contract"], False, "work_report"),
])
def test_kind_table(contract, gate, expected):
    assert kind(contract, gate) == expected


def test_clean_fence_framing():
    assert norm(' {"a": 1} ') == ('{"a": 1}', None)
    assert norm('```json\n{"a": 1}\n```') == ('{"a": 1}', "fence")
    assert norm('Here you go: {"a": {"b": 2}} done.') == ('{"a": {"b": 2}}', "framing")


def test_braces_inside_strings():
    assert norm('Result: {"a": "x } { y"}') == ('{"a": "x } { y"}', "framing")


@pytest.mark.parametrize("raw", [
    '{"a": 1} {"b": 2}',
    '{"a": 1, "b":',
    '{"a": 1, "a": 2}',
    '{"a": {"b": 1, "b": 2}}',
    'see {...} then {"a": 1}',
    'no json here',
    '```json\nnote {"a": 1}\n```',
])
def test_failures(raw):
    with pytest.raises(ValueError):
        norm(raw)
