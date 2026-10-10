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
from huddleroom.models.orchestration import delegable_context
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
from huddleroom.services.orchestration_service import OrchestrationService
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

# Unit tests of _classify/resolve_creation_shape must carry real_shape, or the autouse conftest stub replaces _classify.
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
        raise GoalShapeResolutionError("provider_error", "upstream-secret-text")

    with patch.object(OrchestrationGoalShapeService, "_classify", fail):
        resp = await client.post(f"/api/v1/projects/{test_project.id}/orchestration/goals", json=_api_payload())
    assert resp.status_code == 503
    assert resp.json()["detail"]["code"] == "goal_shape_unresolved"
    assert resp.json()["detail"]["reason"] == "provider_error"
    assert "upstream-secret-text" not in resp.text
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


# --- Task 4 tester additions ---

@real_shape
def test_conftest_stub_skipped_for_real_goal_shape_marker():
    assert OrchestrationGoalShapeService._classify.__module__ == "huddleroom.services.orchestration_goal_shape"


def test_conftest_stub_active_for_unmarked_tests():
    assert OrchestrationGoalShapeService._classify.__module__ != "huddleroom.services.orchestration_goal_shape"


@pytest.mark.parametrize("reason", ["provider_error", "invalid_response"])
async def test_api_unresolved_shape_reports_reason_and_creates_nothing(client, db_session, test_project, reason):
    async def fail(self, project_id, data):
        raise GoalShapeResolutionError(reason, "bad")

    with patch.object(OrchestrationGoalShapeService, "_classify", fail):
        resp = await client.post(f"/api/v1/projects/{test_project.id}/orchestration/goals", json=_api_payload())
    assert resp.status_code == 503
    assert resp.json()["detail"]["reason"] == reason
    assert (await db_session.execute(select(OrchestrationGoal))).all() == []
    assert (await db_session.execute(select(OrchestrationRun))).all() == []


async def test_api_weight_is_independent_of_resolved_shape(client, test_project):
    url = f"/api/v1/projects/{test_project.id}/orchestration/goals"
    payload = _api_payload(budget={"caps": {"max_hours": 200}}, success_criteria=[
        {"key": f"c{i}", "description": f"criterion {i}"} for i in range(8)])
    weights = {
        gt: (await client.post(url, json={**payload, "goal_type": gt})).json()["goal"]["weight"]
        for gt in ("auto", "outcome", "roadmap", "continuous")
    }
    assert len(set(weights.values())) == 1, weights


@pytest.mark.parametrize("goal_type", ["outcome", "roadmap", "continuous"])
async def test_api_explicit_create_persists_rationales_and_policy(client, test_project, goal_type):
    resp = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals", json=_api_payload(goal_type=goal_type))
    goal = resp.json()["goal"]
    assert goal["goal_type_rationale"] == "Selected by user at goal creation."
    assert goal["planning_rationale"] == f"Default planning mode for an explicitly selected {goal_type} goal."
    assert goal["orchestrator_context"]["expansion_policy"] == _DEFAULT_POLICY


@real_shape
async def test_classifier_request_redacts_secrets():
    secret = "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
    fake = scripted(shape_reply("outcome", "delivery"))
    data = create("Ship it", constraints={"note": f"use key {secret}"})
    await OrchestrationGoalShapeService(fake).resolve_creation_shape(PROJECT_ID, data)
    assert secret not in json.dumps(fake.calls[0]["messages"])


def test_delegable_context_drops_only_expansion_policy():
    ctx = {"expansion_policy": {"depth": 0}, "team": {"a": 1}}
    assert delegable_context(ctx) == {"team": {"a": 1}}
    assert "expansion_policy" in ctx and delegable_context(None) == {}


def test_planning_contract_excludes_expansion_policy():
    from types import SimpleNamespace

    goal = SimpleNamespace(
        id=uuid.uuid4(), objective="Ship", success_criteria=[], constraints={}, budget={}, goal_type="outcome",
        planning_mode="implementation",
        orchestrator_context={"expansion_policy": {"depth": 0}, "team": {"agent_ids": ["x"]}},
    )
    request = {"agent_id": str(uuid.uuid4()), "scope": "plan it"}
    contract = OrchestrationService()._planning_contract(
        goal, SimpleNamespace(id=uuid.uuid4(), planning_mode_snapshot=None, plan_state={}), SimpleNamespace(id=uuid.uuid4()), request,
    )
    assert contract.orchestrator_context == {"team": {"agent_ids": ["x"]}}


# --- Task 14c: correction preview, retained manifest, preview fingerprint ---

from datetime import datetime, timezone
import hashlib

from sqlalchemy import event as sa_event, func

from huddleroom.models.artifact import Artifact
from huddleroom.models.event_log import EventLog
from huddleroom.models.orchestration import (
    OrchestrationAction,
    OrchestrationBudgetReservation,
    OrchestrationDecision,
    OrchestrationEvidence,
    OrchestrationGate,
    OrchestrationWait,
    compute_goal_contract_fingerprint,
)
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.services.orchestration_budget_service import BudgetMeasurementError, OrchestrationBudgetService
from tests.test_orchestration_continuous_service import direct_policy


async def _goal_run(db, project, *, started=True, parent=None, **kw):
    lineage = {} if parent is None else dict(
        parent_goal_id=parent.id, continuous_origin_key=f"case-{uuid.uuid4()}", parent_contract_snapshot={},
        goal_delta={},
    )
    goal = OrchestrationGoal(**{
        "project_id": project.id, "objective": "Ship it", "original_request": "Ship it", "success_criteria": [],
        "constraints": {}, "budget": {}, "goal_type_source": "user", **lineage, **kw,
    })
    db.add(goal)
    await db.flush()
    run = OrchestrationRun(
        goal_id=goal.id, plan_state={}, active_blockers=[], budget_state={}, retry_state={},
        goal_type_snapshot=goal.goal_type, planning_mode_snapshot=goal.planning_mode,
        goal_contract_fingerprint=compute_goal_contract_fingerprint(goal) if started else None,
        phase="authorized" if started else "baseline",
    )
    db.add(run)
    await db.flush()
    return goal, run


async def _preview(db, goal, run, actor="human:u1", **kw):
    request = GoalTypeCorrectionPreviewRequest(**{
        "requested_goal_type": "roadmap", "expected_run_id": run.id,
        "expected_contract_fingerprint": compute_goal_contract_fingerprint(goal), "rationale": "fits better", **kw,
    })
    return await OrchestrationGoalShapeService().preview_correction(db, goal, request, actor)


def _codes(result):
    return [blocker.code for blocker in result.blockers]


async def _run_task(db, project, run, status="done", **orchestration):
    task = Task(project_id=project.id, title="work", status=status,
                metadata_={"orchestration": {"run_id": str(run.id), **orchestration}})
    db.add(task)
    await db.flush()
    return task


async def _accepted_gate(db, run, gate_type="work_completed", **kw):
    gate = OrchestrationGate(run_id=run.id, success_criterion_key="done", gate_type=gate_type,
                             required_evidence=kw.pop("required_evidence", {}), status=kw.pop("status", "accepted"))
    db.add(gate)
    await db.flush()
    return gate


async def test_preview_clean_root(db_session, test_project):
    goal, run = await _goal_run(db_session, test_project, goal_type="outcome", budget={"caps": {"max_tokens": "1000"}})
    result = await _preview(db_session, goal, run)
    assert result.blockers == [] and result.retained_evidence_manifest == []
    assert (result.goal_id, result.current_run_id) == (goal.id, run.id)
    assert (result.current_goal_type, result.current_planning_mode) == ("outcome", goal.planning_mode)
    assert (result.proposed_goal_type, result.proposed_planning_mode, result.source) == ("roadmap", "roadmap", "user")
    assert result.remaining_budget == {"max_tokens": Decimal("1000")}
    assert result.requires_new_start is True and result.missing_recurring_policy is False
    assert result.required_authority == ["baseline_rerun", "start"]


async def test_preview_stale_terminal_same_type_blockers_sorted(db_session, test_project):
    goal, run = await _goal_run(db_session, test_project, goal_type="outcome")
    goal.status = "completed"
    result = await _preview(db_session, goal, run, requested_goal_type="outcome",
                            expected_run_id=uuid.uuid4(), expected_contract_fingerprint=FP)
    assert _codes(result) == ["goal_terminal", "same_goal_type", "stale_contract", "stale_run"]
    assert result.blockers[-1].entity_id == run.id


async def test_auto_recommendation_for_user_goal_requires_user(db_session, test_project):
    goal, run = await _goal_run(db_session, test_project)
    assert _codes(await _preview(db_session, goal, run, actor="orchestrator")) == ["explicit_choice_requires_user"]
    assert _codes(await _preview(db_session, goal, run, actor="human:u1")) == []
    goal.goal_type_source = "auto"
    assert _codes(await _preview(db_session, goal, run, actor="orchestrator")) == []


async def test_preview_child_blockers_and_authority(db_session, test_project):
    parent, _ = await _goal_run(db_session, test_project)
    child, child_run = await _goal_run(db_session, test_project, parent=parent)
    child.goal_type_source = "parent_explicit"
    result = await _preview(db_session, child, child_run, requested_goal_type="continuous")
    assert _codes(result) == [
        "continuous_child_requires_continuous_root", "parent_explicit_requires_parent_plan",
        "parent_reservation_inactive",
    ]
    assert result.blockers[-1].entity_id is None
    assert result.requires_new_start is False and result.required_authority == ["parent_plan", "authority_review"]
    assert result.missing_recurring_policy is True

    child.goal_type_source = "parent_auto"
    reservation = OrchestrationBudgetReservation(
        parent_goal_id=parent.id, child_goal_id=child.id, continuous_origin_key="case-1", allocation={},
    )
    db_session.add(reservation)
    await db_session.flush()
    result = await _preview(db_session, child, child_run)
    assert result.blockers == [] and result.required_authority == ["parent_release"]

    reservation.status, reservation.settled_at, reservation.settlement_reason = "settled", datetime.now(timezone.utc), "completed"
    await db_session.flush()
    result = await _preview(db_session, child, child_run)
    assert [(b.code, b.entity_id) for b in result.blockers] == [("parent_reservation_inactive", reservation.id)]

    parent_run = await OrchestrationService().get_run_for_goal(db_session, test_project.id, parent.id)
    assert [(b.code, b.entity_id) for b in (await _preview(db_session, parent, parent_run)).blockers] == [
        ("active_child_goal", child.id),
    ]


async def test_preview_active_work_blockers(db_session, test_project, test_agent):
    goal, run = await _goal_run(db_session, test_project)
    running = await _run_task(db_session, test_project, run, status="in_progress")
    ready = await _run_task(db_session, test_project, run, status="ready")
    await _run_task(db_session, test_project, run, status="backlog")
    session = Session(project_id=test_project.id, task_id=running.id, agent_id=test_agent.id,
                      adapter_type="api", status="running")
    action = OrchestrationAction(run_id=run.id, idempotency_key="k", action_type="delegate_task", status="reserved")
    db_session.add_all([session, action])
    await db_session.flush()
    result = await _preview(db_session, goal, run)
    assert {(b.code, b.entity_id) for b in result.blockers} == {
        ("active_session", session.id), ("active_task", running.id), ("active_task", ready.id),
        ("reserved_action", action.id),
    }
    assert result.blockers == sorted(result.blockers, key=lambda b: (b.code, str(b.entity_id)))


async def test_preview_budget_measurement_blocker(db_session, test_project, monkeypatch):
    goal, run = await _goal_run(db_session, test_project)
    session_id = uuid.uuid4()

    async def unmeasured(self, db, parent):
        raise BudgetMeasurementError("max_tokens", session_id)

    monkeypatch.setattr(OrchestrationBudgetService, "remaining", unmeasured)
    result = await _preview(db_session, goal, run)
    assert [(b.code, b.entity_id) for b in result.blockers] == [("budget_measurement", session_id)]
    assert result.remaining_budget == {}


async def test_preview_recurring_policy_flag(db_session, test_project):
    goal, run = await _goal_run(db_session, test_project)
    result = await _preview(db_session, goal, run, requested_goal_type="continuous")
    assert result.missing_recurring_policy is True and result.blockers == []
    assert result.required_authority == ["baseline_rerun", "start", "continuous_policy"]
    goal.continuous_policy = {"version": 1, **direct_policy()}
    result = await _preview(db_session, goal, run, requested_goal_type="continuous")
    assert result.missing_recurring_policy is False and result.required_authority == ["baseline_rerun", "start"]


async def _counts(db):
    tables = (OrchestrationGoal, OrchestrationRun, OrchestrationGate, OrchestrationEvidence, OrchestrationAction,
              OrchestrationDecision, OrchestrationWait, OrchestrationBudgetReservation, Task, Session, EventLog, Artifact)
    return {table.__name__: await db.scalar(select(func.count()).select_from(table)) for table in tables}


async def test_preview_writes_nothing(db_session, test_project, test_agent):
    goal, run = await _goal_run(db_session, test_project, budget={"caps": {"max_tokens": "1000"}})
    gate = await _accepted_gate(db_session, run)
    task = await _run_task(db_session, test_project, run, status="in_progress", gate_id=str(gate.id))
    db_session.add(Session(project_id=test_project.id, task_id=task.id, agent_id=test_agent.id,
                           adapter_type="api", status="running"))
    await db_session.flush()
    before = await _counts(db_session)
    statements = []

    def record(conn, cursor, statement, *args):
        statements.append(statement)

    engine = db_session.bind.sync_engine
    sa_event.listen(engine, "before_cursor_execute", record)
    try:
        result = await _preview(db_session, goal, run, requested_goal_type="continuous")
    finally:
        sa_event.remove(engine, "before_cursor_execute", record)
    assert result.blockers and result.retained_evidence_manifest
    assert statements and all(sql.lstrip().upper().startswith("SELECT") for sql in statements)
    assert not db_session.new and not db_session.dirty and not db_session.deleted
    assert await _counts(db_session) == before


async def test_preview_fingerprint_deterministic_and_sensitive(db_session, test_project, monkeypatch):
    goal, run = await _goal_run(db_session, test_project)
    first = await _preview(db_session, goal, run)
    assert (await _preview(db_session, goal, run)).preview_fingerprint == first.preview_fingerprint
    payload = {**first.model_dump(mode="json", exclude={"preview_fingerprint"}),
               "contract_fingerprint": compute_goal_contract_fingerprint(goal)}
    assert first.preview_fingerprint == hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()

    assert (await _preview(db_session, goal, run, rationale="other")).preview_fingerprint != first.preview_fingerprint

    async def richer(self, db, parent):
        return {"remaining": {"max_tokens": "5"}}

    with monkeypatch.context() as patched:
        patched.setattr(OrchestrationBudgetService, "remaining", richer)
        assert (await _preview(db_session, goal, run)).preview_fingerprint != first.preview_fingerprint

    db_session.add(OrchestrationAction(run_id=run.id, idempotency_key="k", action_type="x", status="reserved"))
    await db_session.flush()
    assert (await _preview(db_session, goal, run)).preview_fingerprint != first.preview_fingerprint


async def test_manifest_contract_unchanged_retains_without_revalidation(db_session, test_project):
    goal, run = await _goal_run(db_session, test_project)
    gate = await _accepted_gate(db_session, run)
    task = await _run_task(db_session, test_project, run, gate_id=str(gate.id))
    adhoc = await _accepted_gate(db_session, run, gate_type="adhoc_work_accepted")
    adopted = await _run_task(db_session, test_project, run, adhoc_gate_id=str(adhoc.id))
    open_gate = await _accepted_gate(db_session, run, status="open")
    await _run_task(db_session, test_project, run, gate_id=str(open_gate.id))
    artifact = Artifact(project_id=test_project.id, name="spec", artifact_type="document")
    db_session.add(artifact)
    await db_session.flush()
    db_session.add(OrchestrationEvidence(run_id=run.id, gate_id=gate.id, source_type="artifact",
                                         source_id=artifact.id, verdict="accepted"))
    await db_session.flush()

    manifest = (await _preview(db_session, goal, run)).retained_evidence_manifest
    assert [(e.source_gate_id, e.source_task_id) for e in manifest] == sorted(
        [(gate.id, task.id), (adhoc.id, adopted.id)], key=lambda pair: str(pair[0]))
    by_gate = {e.source_gate_id: e for e in manifest}
    assert by_gate[gate.id].artifact_id == artifact.id and by_gate[adhoc.id].artifact_id is None
    assert all(e.source_run_id == run.id and e.source_contract_fingerprint == run.goal_contract_fingerprint
               and e.revalidation_required is False and e.reason.startswith("contract unchanged") for e in manifest)

    goal.objective = "Ship something else"
    manifest = (await _preview(db_session, goal, run)).retained_evidence_manifest
    assert len(manifest) == 2 and all(e.revalidation_required for e in manifest)


async def test_manifest_legacy_null_fingerprint_and_missing_artifact_require_revalidation(db_session, test_project):
    goal, run = await _goal_run(db_session, test_project, started=False)
    gate = await _accepted_gate(db_session, run)
    await _run_task(db_session, test_project, run, gate_id=str(gate.id))
    [entry] = (await _preview(db_session, goal, run)).retained_evidence_manifest
    assert entry.revalidation_required is True and entry.reason.startswith("requirements changed")
    assert entry.source_contract_fingerprint == compute_goal_contract_fingerprint(goal)

    goal2, run2 = await _goal_run(db_session, test_project)
    gate2 = await _accepted_gate(db_session, run2)
    await _run_task(db_session, test_project, run2, gate_id=str(gate2.id))
    db_session.add(OrchestrationEvidence(run_id=run2.id, gate_id=gate2.id, source_type="artifact",
                                         source_id=uuid.uuid4(), verdict="accepted"))
    await db_session.flush()
    [entry] = (await _preview(db_session, goal2, run2)).retained_evidence_manifest
    assert entry.revalidation_required is True


async def test_manifest_retained_gate_carries_original_source(db_session, test_project):
    goal, run = await _goal_run(db_session, test_project)
    source = {
        "source_run_id": str(uuid.uuid4()), "source_task_id": str(uuid.uuid4()), "source_gate_id": str(uuid.uuid4()),
        "artifact_id": None, "source_contract_fingerprint": FP, "revalidation_required": False, "reason": "r",
    }
    await _accepted_gate(db_session, run, gate_type="retained_evidence", required_evidence={**source, "min_count": 0})
    [entry] = (await _preview(db_session, goal, run)).retained_evidence_manifest
    assert (str(entry.source_run_id), str(entry.source_task_id), str(entry.source_gate_id)) == (
        source["source_run_id"], source["source_task_id"], source["source_gate_id"])
    assert entry.source_contract_fingerprint == FP and entry.revalidation_required is False


async def test_pre_start_and_type_specific_lineage(db_session, test_project):
    service = OrchestrationGoalShapeService()
    goal, run = await _goal_run(db_session, test_project, started=False)
    assert await service.is_pre_start(db_session, goal, run) is True
    run.plan_state = {"planning_version": 1}
    assert await service.has_type_specific_lineage(db_session, goal, run) is True
    assert await service.is_pre_start(db_session, goal, run) is False
    run.plan_state = {}
    await _goal_run(db_session, test_project, parent=goal)
    assert await service.has_type_specific_lineage(db_session, goal, run) is True

    other, other_run = await _goal_run(db_session, test_project, started=False)
    db_session.add(OrchestrationAction(run_id=other_run.id, idempotency_key="a", action_type="authorize_execution",
                                       status="completed"))
    await db_session.flush()
    assert await service.has_type_specific_lineage(db_session, other, other_run) is False
    assert await service.is_pre_start(db_session, other, other_run) is False
