import json
import uuid
from decimal import Decimal
from typing import get_args

from unittest.mock import patch

import pytest
from pydantic import ValidationError
from sqlalchemy import select

from huddleroom.config import settings
from huddleroom.models.orchestration import (
    EXPLICIT_PLANNING_MODE_BY_GOAL_TYPE,
    GOAL_TYPE_SOURCE_VALUES,
    GOAL_TYPE_VALUES,
    LEGACY_PLANNING_MODE_BY_GOAL_TYPE,
    PLANNING_MODE_VALUES,
    OrchestrationGoal,
    OrchestrationRun,
)
from huddleroom.schemas import orchestration as s
from huddleroom.schemas.orchestration import (
    GoalShapeDecision,
    GoalTypeCorrectionApplyRequest,
    GoalTypeCorrectionPreviewRequest,
    GoalTypeCorrectionPreviewResult,
    OrchestrationGoalCreate,
    OrchestrationSupersedeRequest,
)
from huddleroom.services.orchestration_goal_definition import classify_goal_weight
from huddleroom.services.orchestration_goal_shape import GoalShapeResolutionError, OrchestrationGoalShapeService

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


def test_preview_result_rejects_more_malformed_nested_items():
    base = preview_result()
    blocker = base["blockers"][0]
    retained = base["retained_evidence_manifest"][0]
    no_artifact = {k: v for k, v in retained.items() if k != "artifact_id"}
    cases = [
        dict(retained_evidence_manifest=[no_artifact]),
        dict(retained_evidence_manifest=[{**retained, "source_contract_fingerprint": "A" * 64}]),
        dict(retained_evidence_manifest=[{**retained, "reason": ""}]),
        dict(retained_evidence_manifest=[{**retained, "source_run_id": "not-a-uuid"}]),
        dict(blockers=[{**blocker, "code": ""}]),
        dict(blockers=[{**blocker, "resolution": ""}]),
        dict(current_goal_type="auto"),
        dict(proposed_goal_type="bogus"),
        dict(source="manual"),
        dict(preview_fingerprint="short"),
        dict(extra="x"),
    ]
    for kw in cases:
        with pytest.raises(ValidationError):
            GoalTypeCorrectionPreviewResult(**{**base, **kw})


def test_no_response_or_decision_model_accepts_auto_goal_type():
    for model in (GoalShapeDecision, GoalTypeCorrectionPreviewRequest):
        field = "goal_type" if model is GoalShapeDecision else "requested_goal_type"
        assert "auto" not in get_args(model.model_fields[field].annotation)


# --- Task 4a: shape resolution service (unit, injected completion_fn) ---

real_shape = pytest.mark.real_goal_shape
PROJECT_ID = uuid.uuid4()


def shape_reply(goal_type, planning_mode, **extra):
    payload = {
        "goal_type": goal_type, "goal_type_rationale": "Because structure.",
        "planning_mode": planning_mode, "planning_rationale": "Because mode.", **extra,
    }
    return {"choices": [{"message": {"content": json.dumps(payload)}}]}


def scripted(*replies):
    calls = []

    async def fake(**request):
        calls.append(request)
        reply = replies[min(len(calls), len(replies)) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply

    fake.calls = calls
    return fake


def create(objective, **kw):
    return OrchestrationGoalCreate(objective=objective, **kw)


def test_create_schema_goal_type_defaults_to_auto():
    assert create("x").goal_type == "auto"
    with pytest.raises(ValidationError):
        create("x", goal_type="epic")


@real_shape
@pytest.mark.parametrize("data,reply,expected", [
    (create("Add CSV export to the reports page"), ("outcome", "delivery"), ("outcome", "delivery")),
    (create("Fix off-by-one in pagination helper"), ("outcome", "implementation"), ("outcome", "implementation")),
    (create("Launch v2: auth migration, then billing, then mobile app"), ("roadmap", "roadmap"), ("roadmap", "roadmap")),
    (create("Every morning triage new GitHub issues and label them"),
     ("continuous", "continuous_policy"), ("continuous", "continuous_policy")),
    (create("Rewrite payments service", success_criteria=[{"key": f"c{i}", "text": f"criterion {i}"} for i in range(8)],
            budget={"caps": {"max_hours": 200}}), ("outcome", "delivery"), ("outcome", "delivery")),
])
async def test_auto_golden(data, reply, expected):
    fake = scripted(shape_reply(*reply))
    result = await OrchestrationGoalShapeService(fake).resolve_creation_shape(PROJECT_ID, data)
    assert (result.goal_type, result.planning_mode) == expected
    assert result.goal_type_source == "auto"
    assert len(fake.calls) == 1
    system = fake.calls[0]["messages"][0]["content"]
    assert data.objective in system and "structure only" in system


@real_shape
async def test_large_bounded_result_weight_is_independent_of_shape():
    data = create(
        "Rewrite payments service", success_criteria=[{"key": f"c{i}", "text": f"criterion {i}"} for i in range(8)],
        budget={"caps": {"max_hours": 200}},
    )
    await OrchestrationGoalShapeService(scripted(shape_reply("outcome", "delivery"))).resolve_creation_shape(PROJECT_ID, data)
    assert classify_goal_weight(data.success_criteria, data.constraints, data.budget, data.objective) == "substantial"


@real_shape
@pytest.mark.parametrize("goal_type", ["outcome", "roadmap", "continuous"])
async def test_explicit_bypasses_model(goal_type):
    async def forbidden(**request):
        raise AssertionError("model must not be called for explicit goal_type")

    result = await OrchestrationGoalShapeService(forbidden).resolve_creation_shape(
        PROJECT_ID, create("anything", goal_type=goal_type))
    assert result.goal_type == goal_type
    assert result.goal_type_source == "user"
    assert result.planning_mode == EXPLICIT_PLANNING_MODE_BY_GOAL_TYPE[goal_type]
    assert result.goal_type_rationale == "Selected by user at goal creation."
    assert result.planning_rationale == f"Default planning mode for an explicitly selected {goal_type} goal."


@real_shape
async def test_misfit_mode_is_repaired():
    fake = scripted(shape_reply("roadmap", "delivery"), shape_reply("roadmap", "roadmap"))
    result = await OrchestrationGoalShapeService(fake).resolve_creation_shape(PROJECT_ID, create("Launch v2"))
    assert (result.goal_type, result.planning_mode) == ("roadmap", "roadmap")
    assert len(fake.calls) == 2
    assert "rejected" in fake.calls[1]["messages"][-1]["content"]


@real_shape
@pytest.mark.parametrize("bad", [
    shape_reply("outcome", "delivery", extra="nope"),
    shape_reply("auto", "delivery"),
])
async def test_invalid_response_after_three_attempts(bad):
    fake = scripted(bad)
    with pytest.raises(GoalShapeResolutionError) as exc:
        await OrchestrationGoalShapeService(fake).resolve_creation_shape(PROJECT_ID, create("x"))
    assert exc.value.reason == "invalid_response"
    assert len(fake.calls) == 3


@real_shape
async def test_provider_error_is_not_retried():
    fake = scripted(RuntimeError("provider down"))
    with pytest.raises(GoalShapeResolutionError) as exc:
        await OrchestrationGoalShapeService(fake).resolve_creation_shape(PROJECT_ID, create("x"))
    assert exc.value.reason == "provider_error"
    assert len(fake.calls) == 1


# --- API wiring (autouse stub replaces _classify unless a test patches it) ---

def _api_payload(**extra):
    return {
        "objective": "Ship it",
        "success_criteria": [{"key": "done", "description": "Done"}],
        "constraints": {},
        "budget": {"caps": {"max_tokens": 1000}},
        **extra,
    }


async def test_api_default_create_resolves_auto(client, test_project):
    resp = await client.post(f"/api/v1/projects/{test_project.id}/orchestration/goals", json=_api_payload())
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["goal"]["goal_type"] in GOAL_TYPE_VALUES
    assert (body["goal"]["goal_type"], body["goal"]["goal_type_source"], body["goal"]["planning_mode"]) == (
        "outcome", "auto", "implementation",
    )
    assert (body["run"]["goal_type_snapshot"], body["run"]["planning_mode_snapshot"]) == ("outcome", "implementation")
    assert body["goal"]["weight"] == classify_goal_weight(
        [{"key": "done", "description": "Done"}], {}, {"caps": {"max_tokens": 1000}}, "Ship it",
    )


@pytest.mark.parametrize("goal_type", ["outcome", "roadmap", "continuous"])
async def test_api_explicit_create_bypasses_classifier(client, test_project, goal_type):
    async def boom(self, project_id, data):
        raise AssertionError("classifier must not run for explicit goal types")

    with patch.object(OrchestrationGoalShapeService, "_classify", boom):
        resp = await client.post(
            f"/api/v1/projects/{test_project.id}/orchestration/goals", json=_api_payload(goal_type=goal_type),
        )
    assert resp.status_code == 201, resp.text
    goal = resp.json()["goal"]
    assert (goal["goal_type"], goal["goal_type_source"]) == (goal_type, "user")
    assert goal["planning_mode"] == EXPLICIT_PLANNING_MODE_BY_GOAL_TYPE[goal_type]


async def test_api_unknown_goal_type_is_422(client, test_project):
    resp = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals", json=_api_payload(goal_type="epic"),
    )
    assert resp.status_code == 422


async def test_api_unresolved_shape_is_503_and_creates_nothing(client, db_session, test_project):
    async def fail(self, project_id, data):
        raise GoalShapeResolutionError("provider_error", "down")

    with patch.object(OrchestrationGoalShapeService, "_classify", fail):
        resp = await client.post(f"/api/v1/projects/{test_project.id}/orchestration/goals", json=_api_payload())
    assert resp.status_code == 503
    assert resp.json()["detail"]["code"] == "goal_shape_unresolved"
    assert resp.json()["detail"]["reason"] == "provider_error"
    goals = (await db_session.execute(select(OrchestrationGoal).where(OrchestrationGoal.project_id == test_project.id))).all()
    runs = (await db_session.execute(select(OrchestrationRun))).all()
    assert goals == [] and runs == []


# --- expansion policy freeze ---

_DEFAULT_POLICY = {"max_concurrent_tasks": 2, "max_goal_depth": 3, "max_descendant_goals": 50, "depth": 0}


async def test_api_create_freezes_expansion_policy(client, test_project):
    resp = await client.post(f"/api/v1/projects/{test_project.id}/orchestration/goals", json=_api_payload())
    assert resp.status_code == 201, resp.text
    assert resp.json()["goal"]["orchestrator_context"] == {"expansion_policy": _DEFAULT_POLICY}


async def test_settings_override_applies_only_to_new_goals(client, test_project):
    url = f"/api/v1/projects/{test_project.id}/orchestration/goals"
    first = await client.post(url, json=_api_payload())
    with patch.object(settings, "orchestration_max_concurrent_tasks", 4):
        second = await client.post(url, json=_api_payload())
    assert first.json()["goal"]["orchestrator_context"]["expansion_policy"]["max_concurrent_tasks"] == 2
    assert second.json()["goal"]["orchestrator_context"]["expansion_policy"]["max_concurrent_tasks"] == 4


async def test_reset_preserves_expansion_policy(client, test_project):
    url = f"/api/v1/projects/{test_project.id}/orchestration/goals"
    with patch.object(settings, "orchestration_max_concurrent_tasks", 4):
        goal_id = (await client.post(url, json=_api_payload())).json()["goal"]["id"]
    resp = await client.post(f"{url}/{goal_id}/reset")
    assert resp.status_code == 200, resp.text
    assert resp.json()["goal"]["orchestrator_context"] == {"expansion_policy": {**_DEFAULT_POLICY, "max_concurrent_tasks": 4}}
