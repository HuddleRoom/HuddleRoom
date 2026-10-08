import uuid
from decimal import Decimal
from typing import get_args

import pytest
from pydantic import ValidationError

from huddleroom.models.orchestration import (
    EXPLICIT_PLANNING_MODE_BY_GOAL_TYPE,
    GOAL_TYPE_SOURCE_VALUES,
    GOAL_TYPE_VALUES,
    LEGACY_PLANNING_MODE_BY_GOAL_TYPE,
    PLANNING_MODE_VALUES,
)
from huddleroom.schemas import orchestration as s
from huddleroom.schemas.orchestration import (
    GoalShapeDecision,
    GoalTypeCorrectionApplyRequest,
    GoalTypeCorrectionPreviewRequest,
    GoalTypeCorrectionPreviewResult,
    OrchestrationSupersedeRequest,
)

FP = "a" * 64


def decision(**kw):
    data = dict(
        goal_type="outcome", goal_type_source="user", goal_type_rationale="  why  ",
        planning_mode="delivery", planning_rationale="how",
    )
    return GoalShapeDecision(**{**data, **kw})


def preview_req(**kw):
    data = dict(
        requested_goal_type="roadmap", expected_run_id=str(uuid.uuid4()),
        expected_contract_fingerprint=FP, rationale="because",
    )
    return {**data, **kw}


def preview_result(**kw):
    data = dict(
        goal_id=str(uuid.uuid4()), current_run_id=str(uuid.uuid4()),
        current_goal_type="outcome", proposed_goal_type="roadmap",
        current_planning_mode="delivery", proposed_planning_mode="roadmap",
        source="user", rationale="because",
        blockers=[dict(code="c", entity_type="task", entity_id=None, message="m", resolution="r")],
        retained_evidence_manifest=[dict(
            source_run_id=str(uuid.uuid4()), source_task_id=str(uuid.uuid4()),
            source_gate_id=str(uuid.uuid4()), artifact_id=None,
            source_contract_fingerprint=FP, revalidation_required=True, reason="r",
        )],
        remaining_budget={"max_tokens": "12.5"}, missing_recurring_policy=False,
        requires_new_start=True, required_authority=["owner"], preview_fingerprint=FP,
    )
    return {**data, **kw}


def test_literals_match_model_constants():
    assert get_args(s.GoalType) == GOAL_TYPE_VALUES
    assert get_args(s.PlanningMode) == PLANNING_MODE_VALUES
    assert get_args(s.GoalTypeSource) == GOAL_TYPE_SOURCE_VALUES
    assert get_args(s.GoalTypeSelection) == ("auto", *GOAL_TYPE_VALUES)
    for t in GOAL_TYPE_VALUES:
        assert EXPLICIT_PLANNING_MODE_BY_GOAL_TYPE[t] in s._PLANNING_MODES_BY_GOAL_TYPE[t]
        assert LEGACY_PLANNING_MODE_BY_GOAL_TYPE[t] in s._PLANNING_MODES_BY_GOAL_TYPE[t]


def test_decision_accepts_and_strips():
    assert decision().goal_type_rationale == "why"
    assert decision(planning_mode="implementation").planning_mode == "implementation"


@pytest.mark.parametrize("kw", [
    dict(goal_type="auto"),
    dict(goal_type="roadmap"),
    dict(goal_type="continuous", planning_mode="implementation"),
    dict(goal_type_rationale="  "),
    dict(planning_rationale=""),
    dict(goal_type_source="manual"),
    dict(extra="x"),
])
def test_decision_rejects(kw):
    with pytest.raises(ValidationError):
        decision(**kw)


def test_preview_request():
    assert GoalTypeCorrectionPreviewRequest(**preview_req()).requested_goal_type == "roadmap"


@pytest.mark.parametrize("kw", [
    dict(requested_goal_type="auto"),
    dict(extra="x"),
    dict(rationale="  "),
    dict(expected_contract_fingerprint="z" * 64),
    dict(expected_contract_fingerprint="a" * 63),
])
def test_preview_request_rejects(kw):
    with pytest.raises(ValidationError):
        GoalTypeCorrectionPreviewRequest(**preview_req(**kw))


def test_preview_request_missing_run_id():
    data = preview_req()
    del data["expected_run_id"]
    with pytest.raises(ValidationError):
        GoalTypeCorrectionPreviewRequest(**data)


def test_apply_request():
    GoalTypeCorrectionApplyRequest(**preview_req(preview_fingerprint=FP))
    with pytest.raises(ValidationError):
        GoalTypeCorrectionApplyRequest(**preview_req())
    with pytest.raises(ValidationError):
        GoalTypeCorrectionApplyRequest(**preview_req(preview_fingerprint=FP, extra="x"))


def test_preview_result_valid():
    r = GoalTypeCorrectionPreviewResult(**preview_result())
    assert r.remaining_budget == {"max_tokens": Decimal("12.5")}
    assert r.blockers[0].entity_id is None
    assert r.retained_evidence_manifest[0].artifact_id is None


def test_preview_result_rejects():
    base = preview_result()
    blocker = base["blockers"][0]
    retained = base["retained_evidence_manifest"][0]
    no_entity = {k: v for k, v in blocker.items() if k != "entity_id"}
    cases = [
        dict(blockers=[{**blocker, "extra": 1}]),
        dict(retained_evidence_manifest=[{**retained, "extra": 1}]),
        dict(blockers=[no_entity]),
        dict(proposed_goal_type="auto"),
        dict(proposed_planning_mode="delivery"),
        dict(current_goal_type="continuous"),
    ]
    for kw in cases:
        with pytest.raises(ValidationError):
            GoalTypeCorrectionPreviewResult(**{**base, **kw})


def test_supersede_request_unchanged():
    assert OrchestrationSupersedeRequest().goal_type == "outcome"
    assert set(OrchestrationSupersedeRequest.model_fields) == {"goal_type"}
