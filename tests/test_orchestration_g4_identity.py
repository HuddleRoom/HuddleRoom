"""Retained SPR233 identity contracts; no provider or historic reconstruction."""

import copy
import uuid
from unittest.mock import AsyncMock

import pytest

from huddleroom.models.orchestration_process import OrchestrationProcessRun
from huddleroom.services import orchestration_process_service as service


def _process(snapshot=None):
    return OrchestrationProcessRun(
        id=uuid.uuid4(), goal_id=uuid.uuid4(), process_type="team_hierarchy",
        process_version=4, status="waiting_decision", trigger_reason="test",
        input_snapshot=snapshot or {}, outputs={"transcript": "retained"},
    )


@pytest.mark.parametrize("snapshot,expected", [
    ({}, None),
    ({"fingerprint": "original"}, "original"),
    ({"fingerprint": "original", "effective_comparison": {"fingerprint": "current"}}, "current"),
    ({"effective_comparison": {"fingerprint": "seed", "source": "copied_seed"}}, "seed"),
])
def test_g4_effective_input_fingerprint_contract(snapshot, expected):
    row = _process(snapshot)
    before = copy.deepcopy(row.input_snapshot)
    assert service.effective_input_fingerprint(row) == expected
    assert row.input_snapshot == before


def test_g4_capture_survives_effective_restamps():
    row = _process()
    service.capture_input_fingerprint(row, "actual-input", 3)
    capture = copy.deepcopy(row.input_snapshot["capture"])
    service.set_effective_input_fingerprint(row, "approved-edit", 4, "approved_edit")
    service.set_effective_input_fingerprint(row, "assignment-refresh", 4, "assignment_refresh")
    assert row.input_snapshot["fingerprint"] == "actual-input"
    assert row.input_snapshot["capture"] == capture
    assert capture == {"source": "captured", "formula_version": 3, "original_available": True}
    assert service.effective_input_fingerprint(row) == "assignment-refresh"
    assert row.outputs == {"transcript": "retained"}


@pytest.mark.parametrize("original", ["predecessor-input", None])
def test_g4_debug_seed_initializes_capture_once(original):
    predecessor = _process({"fingerprint": original} if original else {})
    before = copy.deepcopy(predecessor.input_snapshot)
    successor = _process(service.copied_input_seed(predecessor))
    assert "fingerprint" not in successor.input_snapshot
    service.capture_input_fingerprint(successor, "actual-successor-input", 4)
    service.capture_input_fingerprint(successor, "later-input", 4)
    assert successor.input_snapshot["fingerprint"] == "actual-successor-input"
    assert predecessor.input_snapshot == before


def test_g4_legacy_adjustment_does_not_claim_original_invocation():
    row = _process({"fingerprint": "first-observed"})
    service.set_effective_input_fingerprint(row, "current", 5, "formula_upgrade")
    assert row.input_snapshot["fingerprint"] == "first-observed"
    assert row.input_snapshot["capture"] == {
        "source": "legacy_first_observed", "formula_version": 4, "original_available": False,
    }


def test_g4_missing_legacy_identity_is_not_reconstructed_by_effective_restamp():
    row = _process()
    service.set_effective_input_fingerprint(row, "current-roster", 5, "formula_upgrade")
    assert "fingerprint" not in row.input_snapshot
    assert row.input_snapshot["capture"]["original_available"] is False
    assert service.effective_input_fingerprint(row) == "current-roster"


def test_g4_rerun_intent_preserves_hash_and_feedback_once():
    row = _process({"fingerprint": "actual-input"})
    row.outputs["revision_context"] = {"saved_reason": "Keep this feedback"}
    decision_id = uuid.uuid4()
    service.request_process_rerun(row, "hierarchy_changes_requested", decision_id)
    first = copy.deepcopy(row.input_snapshot)
    service.request_process_rerun(row, "hierarchy_changes_requested", decision_id)
    assert row.input_snapshot == first
    assert row.input_snapshot["fingerprint"] == "actual-input"
    assert row.input_snapshot["rerun_intent"]["source_process_id"] == str(row.id)
    assert row.input_snapshot["rerun_intent"]["source_decision_id"] == str(decision_id)
    assert row.outputs["revision_context"]["saved_reason"] == "Keep this feedback"
    assert row.status == "waiting_decision"


def test_g4_legacy_sentinel_is_unprovable():
    raw = "rerun after hierarchy changes requested"
    row = _process({"fingerprint": raw})
    assert service.effective_input_fingerprint(row) is None
    service.set_effective_input_fingerprint(row, "current", 5, "formula_upgrade")
    assert row.input_snapshot["capture"]["original_available"] is False
    assert raw in str(row.input_snapshot)
    assert row.input_snapshot.get("fingerprint") != "current"


@pytest.mark.asyncio
async def test_g4_current_roster_uses_active_rows_and_preserves_reviewed_snapshot(db_session, test_project):
    from huddleroom.models.agent import Agent
    from huddleroom.models.orchestration import OrchestrationGoal
    from huddleroom.models.orchestration_process import OrchestrationAgentReview
    from huddleroom.services.orchestration_roster_mapper import OrchestrationRosterMapper

    agent = Agent(name=f"g4-current-{uuid.uuid4()}", role="reviewer", provider="old", model="reviewed")
    inactive = Agent(name=f"g4-inactive-{uuid.uuid4()}", role="reviewer", provider="old", model="old", is_active=False)
    db_session.add_all([agent, inactive])
    await db_session.flush()
    reviewed = {"provider": agent.provider, "model": agent.model}
    goal = OrchestrationGoal(project_id=test_project.id, objective="Keep history")
    db_session.add(goal)
    await db_session.flush()
    review = OrchestrationAgentReview(
        goal_id=goal.id, agent_id=agent.id, definition_snapshot=reviewed, fit_summary="Reviewed",
    )
    db_session.add(review)
    agent.provider, agent.model = "current", "native-default"
    await db_session.flush()
    roster = await OrchestrationRosterMapper().current_active_agents(db_session)
    assert inactive.id not in {row.id for row in roster}
    current = next(row for row in roster if row.id == agent.id)
    assert (current.provider, current.model) == ("current", "native-default")
    await db_session.refresh(review)
    assert review.definition_snapshot == {"provider": "old", "model": "reviewed"}


@pytest.mark.asyncio
async def test_g4_current_roster_query_failure_never_returns_history():
    from huddleroom.services.orchestration_roster_mapper import OrchestrationRosterMapper

    db = AsyncMock()
    db.execute.side_effect = RuntimeError("current unavailable")
    with pytest.raises(RuntimeError, match="current unavailable"):
        await OrchestrationRosterMapper().current_active_agents(db)
