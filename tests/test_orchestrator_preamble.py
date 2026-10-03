import pytest
from huddleroom.services.orchestration_llm_decision_adapter import orchestrator_preamble


def test_orchestrator_preamble_platform_sentence_only():
    """Test that no arguments returns only the platform sentence."""
    result = orchestrator_preamble()
    expected = (
        "You are the orchestration control plane of a multi-agent software-delivery platform: "
        "a team of AI agents self-organizes to build and operate software while you make only "
        "the coordination judgments — you never produce the work artifacts yourself."
    )
    assert result == expected


def test_orchestrator_preamble_with_project():
    """Test that project info is appended correctly."""
    result = orchestrator_preamble(
        project={"name": "Acme", "description": "widgets"}
    )
    assert "Project: Acme — widgets" in result
    assert result.count("\n") == 1


def test_orchestrator_preamble_with_goal_and_weight():
    """Test that goal and weight are included in the output."""
    result = orchestrator_preamble(
        goal={"objective": "ship X", "weight": "heavy"}
    )
    assert "Goal: ship X" in result
    assert "heavy" in result
    assert "weight:" in result


def test_orchestrator_preamble_with_meeting():
    """Test that meeting info is formatted correctly."""
    result = orchestrator_preamble(
        meeting={"title": "Sprint", "meeting_type": "standup"}
    )
    assert "Meeting: Sprint (standup)" in result


def test_orchestrator_preamble_empty_project_dict():
    """Test that empty project dict does not raise."""
    result = orchestrator_preamble(project={})
    expected = (
        "You are the orchestration control plane of a multi-agent software-delivery platform: "
        "a team of AI agents self-organizes to build and operate software while you make only "
        "the coordination judgments — you never produce the work artifacts yourself."
    )
    assert result == expected


def test_orchestrator_preamble_goal_only_objective():
    """Test goal with only objective field."""
    result = orchestrator_preamble(goal={"objective": "build feature"})
    assert "Goal: build feature" in result
    assert "weight:" not in result
    assert "status:" not in result


def test_orchestrator_preamble_goal_with_status():
    """Test goal with objective and status."""
    result = orchestrator_preamble(
        goal={"objective": "fix bug", "status": "in progress"}
    )
    assert "Goal: fix bug" in result
    assert "status:" in result


def test_orchestrator_preamble_goal_and_meeting_mutual_exclusion():
    """Test that meeting takes precedence when both provided (caller convention)."""
    result = orchestrator_preamble(
        goal={"objective": "ship X"},
        meeting={"title": "Sprint", "meeting_type": "standup"}
    )
    # goal should not be included since meeting is checked in elif
    assert "Goal:" not in result
    assert "Meeting: Sprint (standup)" in result


def test_orchestrator_preamble_project_with_empty_description():
    """Test project with empty description omits the — separator from project line."""
    result = orchestrator_preamble(
        project={"name": "TestProj", "description": ""}
    )
    # Should have the project line without description
    assert "Project: TestProj\n" in result or result.endswith("Project: TestProj")
