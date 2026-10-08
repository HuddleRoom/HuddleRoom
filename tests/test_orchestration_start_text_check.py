import pytest

from huddleroom.services.orchestration_start_text_check import find_start_contradictions


def test_detects_prepare_only():
    text = (
        "Permanent application stack and provider selection remain open. "
        "Prepare only; do not start the goal."
    )
    found = find_start_contradictions({"description": text})
    assert ("description", "Prepare only") in found
    assert ("description", "do not start") in found


@pytest.mark.parametrize(
    "text",
    [
        "do not start the goal",
        "don't start yet",
        "Don't begin yet",
        "never launch",
        "do not yet start",
        "Hold off on execution",
    ],
)
def test_detects_do_not_start_variants(text):
    found = find_start_contradictions({"src": text})
    assert len(found) == 1
    assert found[0][0] == "src"


@pytest.mark.parametrize(
    "text",
    [
        "Start with the API",
        "Run tests before release",
        "Do not publish without approval",
        "prepare the onboarding",
        "do not run migrations in prod",
        "never run as root",
        "don't forget to run tests",
        "never forget to execute the linter",
        "do not want to begin with X",
        "don’t run it",
    ],
)
def test_ignores_unrelated_text(text):
    assert find_start_contradictions({"src": text}) == []


def test_none_sources_ok():
    assert find_start_contradictions({}) == []
    assert find_start_contradictions({"a": None, "b": ""}) == []


def test_reports_source_names_in_order():
    sources = {
        "description": "Prepare only.",
        "notes": None,
        "constraints": "Never begin anything before sign-off.",
        "extra": "Hold off.",
    }
    found = find_start_contradictions(sources)
    assert [name for name, _ in found] == ["description", "constraints", "extra"]
    assert found[0] == ("description", "Prepare only")
