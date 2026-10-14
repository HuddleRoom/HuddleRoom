import json
import uuid
from datetime import timedelta

import pytest
from sqlalchemy.orm.attributes import flag_modified
from sqlalchemy import func, inspect as sa_inspect, select

from huddleroom.models.artifact import Artifact
from huddleroom.models.orchestration import (
    OrchestrationAction,
    OrchestrationEvidence,
    OrchestrationGate,
    OrchestrationGoal,
    OrchestrationRun,
    compute_goal_contract_fingerprint,
)
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.schemas.orchestration import AuthorizationResult
from huddleroom.services.orchestration_service import OrchestrationService
from tests.orchestration_wake_helpers import NOOP_WAKE_WHEN

# SQLAlchemy's dynamic function namespace is not statically callable to pylint.
# pylint: disable=not-callable


@pytest.mark.asyncio
async def test_stop_continuous_goal_is_idempotent_and_exposes_stopped_state(
    client, db_session, auth_headers, test_project,
):
    from tests.test_orchestration_continuous_service import continuous_ready

    goal, _ = await continuous_ready(db_session, test_project)
    await OrchestrationService().start_run(
        db_session, test_project.id, goal.id, actor="human:test",
    )
    url = f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/stop"
    first = await client.post(url, json={"reason": "Maintenance"}, headers=auth_headers)
    second = await client.post(url, json={"reason": "Maintenance"}, headers=auth_headers)
    assert first.status_code == second.status_code == 200
    assert first.json()["goal"]["continuous_state"]["health"] == "stopped"
    assert second.json()["goal"]["continuous_state"] == first.json()["goal"]["continuous_state"]

@pytest.mark.asyncio
async def test_get_run_for_goal_returns_latest_run(db_session, test_project):
    from huddleroom.schemas.orchestration import OrchestrationGoalCreate

    service = OrchestrationService()
    goal, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Orchestration API goal",
            success_criteria=[{"key": "done", "description": "Done"}],
        ),
        created_by_user_id=None,
    )

    fetched = await service.get_run_for_goal(db_session, test_project.id, goal.id)
    assert fetched is not None
    assert fetched.id == run.id

    # Wrong project scope returns None
    assert await service.get_run_for_goal(db_session, uuid.uuid4(), goal.id) is None
    # Unknown goal returns None
    assert await service.get_run_for_goal(db_session, test_project.id, uuid.uuid4()) is None


def _goal_payload(objective="Ship it"):
    return {
        "objective": objective,
        "success_criteria": [{"key": "done", "description": "Done"}],
        "constraints": {},
        "budget": {"caps": {"max_tokens": 1000}},
    }


@pytest.mark.asyncio
async def test_create_goal_returns_goal_and_run_with_empty_ledgers(client, test_project):
    resp = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals",
        json=_goal_payload(),
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["goal"]["objective"] == "Ship it"
    assert body["goal"]["status"] == "active"
    assert body["run"] is not None
    assert body["run"]["status"] == "running"
    assert body["run"]["budget_state"] == {"caps": {"max_tokens": 1000}}
    assert body["decisions"] == []
    assert body["actions"] == []
    assert body["gates"] == []
    assert body["evidence"] == []
    assert body["agent_suggestions"] == []
    assert body["timeline"] == []

    processes = await client.get(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{body['goal']['id']}/processes"
    )
    assert processes.status_code == 200, processes.text
    assert processes.json() == []


# Characterization: later planning-hierarchy work may add fields, never drop these.
GOAL_RESPONSE_KEYS = {
    "id", "project_id", "objective", "original_request", "success_criteria", "constraints", "budget",
    "status", "goal_type", "weight", "supersedes_goal_id", "parent_goal_id", "roadmap_item_key",
    "roadmap_version_id", "continuous_policy", "continuous_state", "continuous_origin_key",
    "orchestrator_context", "needs_you_count", "created_at", "updated_at",
    "goal_type_source", "goal_type_rationale", "planning_mode", "planning_rationale",
}
RUN_RESPONSE_KEYS = {
    "id", "goal_id", "status", "phase", "condition", "baseline_authorized", "budget_state",
    "plan_state", "retry_state", "active_blockers", "cycle_key", "started_at", "completed_at",
    "supersedes_run_id", "terminal_reason", "goal_type_snapshot", "planning_mode_snapshot",
    "goal_contract_fingerprint",
}
DETAIL_RESPONSE_KEYS = {
    "goal", "run", "decisions", "actions", "gates", "evidence", "agent_suggestions", "timeline",
    "decisions_count", "actions_count", "gates_count", "evidence_count", "agent_suggestions_count",
    "roadmap_version", "roadmap_items", "children", "budget_summary", "supervision",
    "contract_fingerprint", "runs", "child_goals", "effective_authorization",
    "task_reports", "plan", "authorization_wait", "capacity_wait",
}


@pytest.mark.asyncio
async def test_create_and_detail_responses_keep_existing_fields_and_outcome_type(client, test_project):
    base = f"/api/v1/projects/{test_project.id}/orchestration/goals"
    created = await client.post(base, json=_goal_payload())
    assert created.status_code == 201, created.text
    goal_id = created.json()["goal"]["id"]
    detail = await client.get(f"{base}/{goal_id}")
    assert detail.status_code == 200, detail.text

    for body in (created.json(), detail.json()):
        assert DETAIL_RESPONSE_KEYS <= set(body)
        assert GOAL_RESPONSE_KEYS <= set(body["goal"])
        assert RUN_RESPONSE_KEYS <= set(body["run"])
        assert body["goal"]["goal_type"] == "outcome"
        assert body["goal"]["goal_type_source"] == "auto"
        assert body["goal"]["planning_mode"] == "implementation"
        assert body["goal"]["goal_type_rationale"].strip() and body["goal"]["planning_rationale"].strip()
        assert body["run"]["goal_type_snapshot"] == "outcome"
        assert body["run"]["planning_mode_snapshot"] == "implementation"
        # Still in baseline: not yet frozen, no successor lineage.
        assert body["run"]["supersedes_run_id"] is None
        assert body["run"]["terminal_reason"] is None
        assert body["run"]["goal_contract_fingerprint"] is None
        assert body["goal"]["parent_goal_id"] is None
        assert body["goal"]["supersedes_goal_id"] is None
        assert body["run"]["phase"] == "baseline"
        # Outcome goals carry no roadmap sections.
        assert body["roadmap_version"] is None
        assert body["roadmap_items"] is None
        assert body["children"] is None


def _row(obj):
    return {c.key: getattr(obj, c.key) for c in sa_inspect(type(obj)).column_attrs}


@pytest.mark.asyncio
async def test_create_goal_does_not_mutate_existing_active_goal_or_run(client, db_session, test_project):
    base = f"/api/v1/projects/{test_project.id}/orchestration/goals"
    first = await client.post(base, json=_goal_payload("Existing goal"))
    assert first.status_code == 201, first.text
    existing_goal_id = uuid.UUID(first.json()["goal"]["id"])
    existing_run_id = uuid.UUID(first.json()["run"]["id"])
    goal = await db_session.get(OrchestrationGoal, existing_goal_id)
    run = await db_session.get(OrchestrationRun, existing_run_id)
    goal_before, run_before = _row(goal), _row(run)
    actions_before = (await db_session.scalar(
        select(func.count()).select_from(OrchestrationAction).where(OrchestrationAction.run_id == run.id)
    ))

    second = await client.post(base, json=_goal_payload("Another goal"))
    assert second.status_code == 201, second.text
    assert second.json()["goal"]["id"] != str(existing_goal_id)

    await db_session.refresh(goal)
    await db_session.refresh(run)
    assert _row(goal) == goal_before
    assert _row(run) == run_before
    assert await db_session.scalar(
        select(func.count()).select_from(OrchestrationAction).where(OrchestrationAction.run_id == run.id)
    ) == actions_before


@pytest.mark.asyncio
async def test_create_goal_unknown_project_returns_404(client):
    resp = await client.post(
        f"/api/v1/projects/{uuid.uuid4()}/orchestration/goals",
        json=_goal_payload(),
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_create_goal_with_empty_success_criteria_asks_for_them(client, test_project):
    """An unclear goal is accepted and clarified through
    the goal-definition process, not rejected at intake. Supersedes the
    pre-Phase-5 `test_create_goal_rejects_empty_success_criteria` behavior."""
    payload = _goal_payload()
    payload["success_criteria"] = []
    resp = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals",
        json=payload,
    )
    assert resp.status_code == 201
    assert resp.json()["goal"]["weight"] == "trivial"


@pytest.mark.asyncio
async def test_list_and_get_goal_detail(client, test_project):
    created = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals",
        json=_goal_payload("Listed goal"),
    )
    goal_id = created.json()["goal"]["id"]

    listed = await client.get(f"/api/v1/projects/{test_project.id}/orchestration/goals")
    assert listed.status_code == 200
    body = listed.json()
    goal_item = next(item for item in body["items"] if item["id"] == goal_id)
    assert goal_item["needs_you_count"] == 0

    detail = await client.get(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal_id}"
    )
    assert detail.status_code == 200
    assert detail.json()["goal"]["id"] == goal_id
    assert detail.json()["goal"]["needs_you_count"] == 0
    assert detail.json()["run"]["status"] == "running"


@pytest.mark.asyncio
async def test_goal_detail_returns_human_assignment_and_agent_suggestion_actions(client, db_session, test_project):
    created = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals",
        json=_goal_payload("Goal detail"),
    )
    assert created.status_code == 201, created.text
    goal_id = created.json()["goal"]["id"]
    run_id = uuid.UUID(created.json()["run"]["id"])

    service = OrchestrationService()
    action = await service.execute_suggest_agent_action(
        db_session,
        run_id=run_id,
        request={
            "action_type": "suggest_agent",
            "missing_work_function": "validation",
            "reason": "No active agent can provide independent validation.",
            "suggested_role": "validator",
            "suggested_capabilities": ["validation", "testing"],
            "suggested_adapter_type": "api",
            "suggested_model": "gpt-4o-mini",
            "suggested_system_prompt_outline": "Validate completed work and report evidence without editing artifacts.",
        },
        idempotency_key="run:api-detail:kind:suggest_agent:validation",
    )

    detail = await client.get(f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal_id}")

    assert detail.status_code == 200, detail.text
    body = detail.json()
    assert len(body["actions"]) == 1
    assert body["actions"][0]["id"] == str(action.id)
    assert body["actions"][0]["action_type"] == "suggest_agent"
    assert body["actions"][0]["target_type"] == "agent_suggestion"
    assert body["actions"][0]["target_id"] == str(action.target_id)
    assert len(body["agent_suggestions"]) == 1
    assert body["agent_suggestions"][0]["id"] == str(action.target_id)
    assert body["agent_suggestions"][0]["missing_work_function"] == "validation"
    assert body["agent_suggestions"][0]["suggested_capabilities"] == ["validation", "testing"]


@pytest.mark.asyncio
async def test_goal_detail_returns_complete_audit_ledgers(client, db_session, test_project):
    created = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals",
        json=_goal_payload("Complete audit detail"),
    )
    assert created.status_code == 201, created.text
    goal_id = created.json()["goal"]["id"]
    run_id = uuid.UUID(created.json()["run"]["id"])

    service = OrchestrationService()
    decisions = []
    actions = []
    gates = []
    evidence = []
    for index in range(12):
        decision = await service.record_validated_decision(
            db_session,
            run_id,
            input_snapshot={"index": index},
            llm_output={"action_type": "noop", "reason": f"Reason {index}", "wake_when": NOOP_WAKE_WHEN},
            parsed_decision={"action_type": "noop", "reason": f"Reason {index}", "wake_when": NOOP_WAKE_WHEN},
        )
        decisions.append(decision)
        actions.append(
            await service.execute_suggest_agent_action(
                db_session,
                run_id=run_id,
                request={
                    "action_type": "suggest_agent",
                    "missing_work_function": f"validation_{index}",
                    "reason": f"Missing validator {index}",
                    "suggested_role": "validator",
                    "suggested_capabilities": [f"validation_{index}"],
                },
                idempotency_key=f"run:phase18-detail:kind:suggest_agent:{index}",
                decision_id=decision.id,
            )
        )
        gate = OrchestrationGate(
            run_id=run_id,
            success_criterion_key=f"criterion-{index}",
            gate_type="validation_passed",
            required_evidence={"required_source_types": ["task"], "min_count": 1},
            status="accepted",
        )
        db_session.add(gate)
        await db_session.flush()
        gates.append(gate)
        row = OrchestrationEvidence(
            run_id=run_id,
            gate_id=gate.id,
            source_type="human_override",
            source_id=None,
            verdict="accepted",
            evidence_metadata={"reason": f"Accepted evidence {index}"},
        )
        db_session.add(row)
        evidence.append(row)
    await db_session.flush()

    detail = await client.get(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal_id}"
    )

    assert detail.status_code == 200, detail.text
    body = detail.json()
    assert body["decisions_count"] == 12
    assert {item["id"] for item in body["decisions"]} == {
        str(decision.id) for decision in decisions
    }
    assert [(item["created_at"], item["id"]) for item in body["decisions"]] == sorted(
        (item["created_at"], item["id"]) for item in body["decisions"]
    )
    assert {item["reason"] for item in body["decisions"]} == {
        f"Reason {index}" for index in range(12)
    }
    assert {item["validator_status"] for item in body["decisions"]} == {"accepted"}
    assert body["actions_count"] == 12
    assert {item["id"] for item in body["actions"]} == {
        str(action.id) for action in actions
    }
    assert [(item["created_at"], item["id"]) for item in body["actions"]] == sorted(
        (item["created_at"], item["id"]) for item in body["actions"]
    )
    assert body["gates_count"] == 12
    assert {item["id"] for item in body["gates"]} == {str(gate.id) for gate in gates}
    assert [(item["created_at"], item["id"]) for item in body["gates"]] == sorted(
        (item["created_at"], item["id"]) for item in body["gates"]
    )
    assert body["evidence_count"] == 12
    assert {item["id"] for item in body["evidence"]} == {
        str(row.id) for row in evidence
    }
    assert [(item["created_at"], item["id"]) for item in body["evidence"]] == sorted(
        (item["created_at"], item["id"]) for item in body["evidence"]
    )
    assert body["agent_suggestions_count"] == 12
    assert {item["id"] for item in body["agent_suggestions"]} == {
        str(action.target_id) for action in actions
    }
    assert [
        (item["created_at"], item["id"]) for item in body["agent_suggestions"]
    ] == sorted(
        (item["created_at"], item["id"]) for item in body["agent_suggestions"]
    )


@pytest.mark.asyncio
async def test_goal_detail_without_run_returns_empty_ledgers(client, db_session, test_project):
    created = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals",
        json=_goal_payload("Run-absent audit detail"),
    )
    assert created.status_code == 201, created.text
    goal_id = created.json()["goal"]["id"]
    run_id = uuid.UUID(created.json()["run"]["id"])

    run = await db_session.get(OrchestrationRun, run_id)
    await db_session.delete(run)
    await db_session.flush()

    detail = await client.get(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal_id}"
    )

    assert detail.status_code == 200, detail.text
    body = detail.json()
    assert body["run"] is None
    assert body["meeting_commitments"] == []
    assert body["decisions_count"] == 0
    assert body["decisions"] == []
    assert body["actions_count"] == 0
    assert body["actions"] == []
    assert body["gates_count"] == 0
    assert body["gates"] == []
    assert body["evidence_count"] == 0
    assert body["evidence"] == []
    assert body["agent_suggestions_count"] == 0
    assert body["agent_suggestions"] == []


@pytest.mark.asyncio
async def test_get_unknown_goal_returns_404(client, test_project):
    resp = await client.get(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{uuid.uuid4()}"
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_list_goals_invalid_status_returns_400(client, test_project):
    resp = await client.get(
        f"/api/v1/projects/{test_project.id}/orchestration/goals?status=not-real"
    )
    assert resp.status_code == 400


async def _create_goal_id(client, project_id, objective="Lifecycle goal"):
    resp = await client.post(
        f"/api/v1/projects/{project_id}/orchestration/goals",
        json=_goal_payload(objective),
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["goal"]["id"]


@pytest.mark.asyncio
async def test_pause_resume_cancel_happy_path(client, test_project):
    pid = test_project.id
    goal_id = await _create_goal_id(client, pid)

    paused = await client.post(f"/api/v1/projects/{pid}/orchestration/goals/{goal_id}/pause")
    assert paused.status_code == 200
    assert paused.json()["goal"]["status"] == "paused"
    assert paused.json()["run"]["status"] == "paused"

    resumed = await client.post(f"/api/v1/projects/{pid}/orchestration/goals/{goal_id}/resume")
    assert resumed.status_code == 200
    assert resumed.json()["goal"]["status"] == "active"
    assert resumed.json()["run"]["status"] == "running"

    cancelled = await client.post(f"/api/v1/projects/{pid}/orchestration/goals/{goal_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["goal"]["status"] == "cancelled"
    assert cancelled.json()["run"]["status"] == "cancelled"
    assert cancelled.json()["run"]["completed_at"] is not None


@pytest.mark.asyncio
async def test_list_goals_needs_you_count_aggregates_blockers_gates_decisions_warnings(client, db_session, test_project):
    """Test that needs_you_count aggregates active blockers, failed gates, pending human decisions, and active warnings."""
    from huddleroom.schemas.orchestration import OrchestrationGoalCreate
    from huddleroom.models.orchestration_process import OrchestrationWarning
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
    from datetime import datetime, timezone

    service = OrchestrationService()

    # Create control goal (no blockers, gates, decisions, warnings)
    control_goal, control_run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Control goal",
            success_criteria=[{"key": "done", "description": "Done"}],
        ),
        created_by_user_id=None,
    )

    # Create test goal with various triage items
    test_goal, test_run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Test goal with triage",
            success_criteria=[{"key": "done", "description": "Done"}],
        ),
        created_by_user_id=None,
    )

    # 1. Add active blocker (modifying run's active_blockers JSON list)
    test_run.active_blockers = [{"kind": "task_blocked", "reason": "Waiting for approval"}]
    db_session.add(test_run)

    # 2. Add failed gate
    failed_gate = OrchestrationGate(
        run_id=test_run.id,
        success_criterion_key="done",
        gate_type="validation_passed",
        required_evidence={"required_source_types": ["task"], "min_count": 1},
        status="failed",
        failure_reason="No evidence provided",
    )
    db_session.add(failed_gate)

    # 3. Add active warning
    warning_svc = OrchestrationWarningService()
    await warning_svc.create_warning(
        db_session,
        goal_id=test_goal.id,
        warning_type="missing_capability",
        severity="warning",
        message="Team lacks required capability",
    )

    # 4. Add pending human decision
    decision_svc = OrchestrationAuthorityDecisionService()
    from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
    pending_decision = OrchestrationAuthorityDecision(
        goal_id=test_goal.id,
        run_id=test_run.id,
        decision_key="test-decision",
        title="Test Decision",
        status="pending",
        authority="human",
        question="Should we proceed?",
        options=["yes", "no"],
        asked_at=datetime.now(timezone.utc),
    )
    db_session.add(pending_decision)

    await db_session.flush()

    # Verify list_goals returns correct counts
    items, _ = await service.list_goals(db_session, test_project.id)
    control_item = next(item for item in items if item.id == control_goal.id)
    test_item = next(item for item in items if item.id == test_goal.id)

    assert control_item.needs_you_count == 0
    # Expected: 1 (active blocker) + 1 (failed gate) + 1 (pending decision) + 1 (active warning) = 4
    assert test_item.needs_you_count == 4


@pytest.mark.asyncio
async def test_invalid_transitions_return_409(client, test_project):
    pid = test_project.id
    goal_id = await _create_goal_id(client, pid)

    # Resume an active (not paused) goal → 409
    resp = await client.post(f"/api/v1/projects/{pid}/orchestration/goals/{goal_id}/resume")
    assert resp.status_code == 409

    # Pause twice → second pause is 409
    assert (await client.post(f"/api/v1/projects/{pid}/orchestration/goals/{goal_id}/pause")).status_code == 200
    assert (await client.post(f"/api/v1/projects/{pid}/orchestration/goals/{goal_id}/pause")).status_code == 409

    # Cancel, then any transition on the cancelled goal → 409
    await client.post(f"/api/v1/projects/{pid}/orchestration/goals/{goal_id}/resume")  # back to active
    assert (await client.post(f"/api/v1/projects/{pid}/orchestration/goals/{goal_id}/cancel")).status_code == 200
    assert (await client.post(f"/api/v1/projects/{pid}/orchestration/goals/{goal_id}/cancel")).status_code == 409
    assert (await client.post(f"/api/v1/projects/{pid}/orchestration/goals/{goal_id}/pause")).status_code == 409


@pytest.mark.asyncio
async def test_lifecycle_unknown_goal_returns_404(client, test_project):
    pid = test_project.id
    missing = uuid.uuid4()
    assert (await client.post(f"/api/v1/projects/{pid}/orchestration/goals/{missing}/pause")).status_code == 404
    assert (await client.post(f"/api/v1/projects/{pid}/orchestration/goals/{missing}/resume")).status_code == 404
    assert (await client.post(f"/api/v1/projects/{pid}/orchestration/goals/{missing}/cancel")).status_code == 404


@pytest.mark.asyncio
async def test_continuous_policy_and_start_are_public_and_idempotent(
    client, db_session, auth_headers, test_project,
):
    from tests.test_orchestration_continuous_service import continuous_ready, direct_policy

    goal, run = await continuous_ready(db_session, test_project)
    updated = await client.put(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/continuous-policy",
        json={"policy": direct_policy(response_target_seconds=600)}, headers=auth_headers,
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["goal"]["continuous_policy"]["version"] == 2

    first = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/start", headers=auth_headers,
    )
    second = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/start", headers=auth_headers,
    )
    assert first.status_code == second.status_code == 200
    assert first.json()["run"]["id"] == second.json()["run"]["id"] == str(run.id)
    assert first.json()["run"]["phase"] == second.json()["run"]["phase"] == "waiting_activation"
    assert first.json()["goal"]["continuous_state"] == second.json()["goal"]["continuous_state"]
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id,
        OrchestrationAction.action_type == "authorize_execution",
    )) == 1


def _type_url(project, goal, suffix="type"):
    return f"/api/v1/projects/{project.id}/orchestration/goals/{goal.id}/{suffix}"


async def _type_body(db, goal, run, **kw):
    from huddleroom.models.orchestration import compute_goal_contract_fingerprint

    return {
        "requested_goal_type": "roadmap", "expected_run_id": str(run.id),
        "expected_contract_fingerprint": compute_goal_contract_fingerprint(goal),
        "rationale": "fits better", **kw,
    }


async def _type_preview(client, db, project, goal, run, headers):
    body = await _type_body(db, goal, run)
    resp = await client.post(_type_url(project, goal, "type-preview"), json=body, headers=headers)
    assert resp.status_code == 200, resp.text
    return body, resp.json()


@pytest.mark.asyncio
async def test_goal_type_preview_and_apply_http(client, db_session, auth_headers, test_project):
    from tests.test_orchestration_goal_shape import _goal_run

    goal, run = await _goal_run(db_session, test_project)
    body, preview = await _type_preview(client, db_session, test_project, goal, run, auth_headers)
    assert preview["blockers"] == [] and preview["preview_fingerprint"]
    assert preview["proposed_goal_type"] == "roadmap"

    resp = await client.post(
        _type_url(test_project, goal), json={**body, "preview_fingerprint": preview["preview_fingerprint"]},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    detail = resp.json()
    assert detail["goal"]["goal_type"] == "roadmap"
    assert detail["run"]["id"] != str(run.id)
    assert detail["run"]["supersedes_run_id"] == str(run.id)


@pytest.mark.asyncio
async def test_goal_type_apply_stale_preview_409(client, db_session, auth_headers, test_project):
    from tests.test_orchestration_goal_shape import _goal_run

    goal, run = await _goal_run(db_session, test_project)
    body = await _type_body(db_session, goal, run)
    resp = await client.post(
        _type_url(test_project, goal), json={**body, "preview_fingerprint": "0" * 64},
        headers=auth_headers,
    )
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert detail["conflict"] == "stale_preview" and "preview" in detail


@pytest.mark.asyncio
async def test_goal_type_apply_blocked_409(client, db_session, auth_headers, test_project, test_agent):
    from huddleroom.models.session import Session
    from tests.test_orchestration_goal_shape import _goal_run, _run_task

    goal, run = await _goal_run(db_session, test_project)
    task = await _run_task(db_session, test_project, run, status="in_progress")
    db_session.add(Session(project_id=test_project.id, task_id=task.id, agent_id=test_agent.id,
                           adapter_type="api", status="running"))
    await db_session.flush()
    body, preview = await _type_preview(client, db_session, test_project, goal, run, auth_headers)
    assert preview["blockers"]
    resp = await client.post(
        _type_url(test_project, goal), json={**body, "preview_fingerprint": preview["preview_fingerprint"]},
        headers=auth_headers,
    )
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert detail["conflict"] == "correction_blocked" and detail["blockers"] and "preview" in detail


@pytest.mark.asyncio
async def test_goal_type_unknown_goal_404_and_bad_body_422(client, db_session, auth_headers, test_project):
    from tests.test_orchestration_goal_shape import _goal_run

    goal, run = await _goal_run(db_session, test_project)
    body = await _type_body(db_session, goal, run)
    ghost = type("G", (), {"id": uuid.uuid4()})()
    fp = {"preview_fingerprint": "0" * 64}
    assert (await client.post(_type_url(test_project, ghost, "type-preview"), json=body, headers=auth_headers)).status_code == 404
    assert (await client.post(_type_url(test_project, ghost), json={**body, **fp}, headers=auth_headers)).status_code == 404
    assert (await client.post(_type_url(test_project, goal, "type-preview"), json={}, headers=auth_headers)).status_code == 422
    assert (await client.post(_type_url(test_project, goal), json=body, headers=auth_headers)).status_code == 422


async def _create_detail(client, test_project, objective="Detail goal"):
    base = f"/api/v1/projects/{test_project.id}/orchestration/goals"
    created = await client.post(base, json=_goal_payload(objective))
    assert created.status_code == 201, created.text
    return f"{base}/{created.json()['goal']['id']}", created.json()


@pytest.mark.asyncio
async def test_detail_exposes_contract_fingerprint_and_baseline_authorization(client, db_session, test_project):
    url, body = await _create_detail(client, test_project)
    goal = await db_session.get(OrchestrationGoal, uuid.UUID(body["goal"]["id"]))
    detail = (await client.get(url)).json()

    assert detail["contract_fingerprint"] == compute_goal_contract_fingerprint(goal)
    auth = detail["effective_authorization"]
    assert set(auth) == set(AuthorizationResult.model_fields)
    assert auth["run_id"] == body["run"]["id"]

    goal.status = "cancelled"
    await db_session.flush()
    cancelled = (await client.get(url)).json()
    assert cancelled["effective_authorization"] is None
    assert cancelled["contract_fingerprint"] == compute_goal_contract_fingerprint(goal)


@pytest.mark.asyncio
async def test_detail_without_run_has_no_runs_or_authorization(client, db_session, test_project):
    url, body = await _create_detail(client, test_project)
    await db_session.delete(await db_session.get(OrchestrationRun, uuid.UUID(body["run"]["id"])))
    await db_session.flush()
    detail = (await client.get(url)).json()

    assert detail["runs"] == [] and detail["effective_authorization"] is None
    assert detail["child_goals"] == []
    assert len(detail["contract_fingerprint"]) == 64


@pytest.mark.asyncio
async def test_detail_run_history_labels_versions_and_marks_current(client, db_session, test_project):
    url, body = await _create_detail(client, test_project)
    run1 = await db_session.get(OrchestrationRun, uuid.UUID(body["run"]["id"]))
    # Cancel first: one active run per goal.
    run1.status, run1.terminal_reason = "cancelled", "reclassified"
    await db_session.flush()
    run2 = OrchestrationRun(
        goal_id=run1.goal_id, supersedes_run_id=run1.id, goal_type_snapshot="roadmap",
        planning_mode_snapshot="roadmap", started_at=run1.started_at + timedelta(seconds=5),
    )
    db_session.add(run2)
    await db_session.flush()

    runs = (await client.get(url)).json()["runs"]

    assert [(r["id"], r["version_label"], r["is_current"]) for r in runs] == [
        (str(run1.id), "v1", False), (str(run2.id), "v2", True),
    ]
    assert (runs[0]["status"], runs[0]["terminal_reason"], runs[0]["supersedes_run_id"]) == (
        "cancelled", "reclassified", None)
    assert runs[1]["supersedes_run_id"] == str(run1.id)
    assert (runs[1]["goal_type_snapshot"], runs[1]["planning_mode_snapshot"]) == ("roadmap", "roadmap")


@pytest.mark.asyncio
async def test_detail_evidence_semantics_are_derived_for_legacy_rows(client, db_session, test_project):
    url, body = await _create_detail(client, test_project)
    run_id = uuid.UUID(body["run"]["id"])
    declared = OrchestrationGate(
        run_id=run_id, success_criterion_key="declared", gate_type="validation_passed",
        required_evidence={"evidence_role": "discovery", "review_stage": "preliminary"}, status="open",
    )
    legacy = OrchestrationGate(
        run_id=run_id, success_criterion_key="legacy", gate_type="validation_passed",
        required_evidence={}, status="accepted",
    )
    db_session.add_all([declared, legacy])
    await db_session.flush()
    db_session.add_all([
        OrchestrationEvidence(run_id=run_id, gate_id=declared.id, source_type="task", verdict="candidate",
                              evidence_metadata={}),
        OrchestrationEvidence(run_id=run_id, gate_id=legacy.id, source_type="verification", verdict="accepted",
                              evidence_metadata={}),
    ])
    await db_session.flush()

    rows = {e["gate_id"]: e for e in (await client.get(url)).json()["evidence"]}

    assert {k: rows[str(declared.id)][k] for k in ("evidence_role", "review_stage", "acceptance_verdict")} == {
        "evidence_role": "discovery", "review_stage": "preliminary", "acceptance_verdict": "pending"}
    assert {k: rows[str(legacy.id)][k] for k in ("evidence_role", "review_stage", "acceptance_verdict")} == {
        "evidence_role": "verification", "review_stage": "final", "acceptance_verdict": "accepted"}


@pytest.mark.asyncio
async def test_detail_lists_released_roadmap_child_with_private_key_set(client, db_session, test_project):
    from tests.test_orchestration_roadmap_children import goal_item, release

    _, parent, _, _, _ = await release(db_session, test_project, goal_item("alpha"))
    detail = (await client.get(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{parent.id}")).json()

    assert len(detail["child_goals"]) == 1
    child = detail["child_goals"][0]
    assert set(child) == {
        "id", "objective", "status", "goal_type", "goal_type_source", "planning_mode", "roadmap_item_key",
        "roadmap_version", "current_run_id", "current_run_status", "run_version_label",
    }
    assert child["id"] == detail["children"][0]
    assert (child["roadmap_item_key"], child["roadmap_version"], child["run_version_label"]) == ("alpha", 1, "v1")
    assert child["current_run_status"] == "running" and child["current_run_id"]
    assert detail["runs"] and detail["runs"][0]["is_current"]
    dumped = json.dumps(detail["child_goals"])
    assert "parent_contract_snapshot" not in dumped and "workspace_policy" not in dumped

    # Direct children only: the child's own detail lists no grandchild.
    child_detail = (await client.get(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{child['id']}")).json()
    assert child_detail["child_goals"] == []


PLAN_ITEM_KEYS = {
    "key", "unit_type", "title", "work_function", "agent_id", "depends_on", "artifact_ids", "why_runnable",
    "priority", "adopt_task_id", "adopted", "task_id", "child_goal_id", "gate_id", "state",
}
PLAN_KEYS = {
    "status", "artifact_id", "kind", "author_agent_id", "planning_version", "planning_task_id", "summary",
    "scope", "items",
}


@pytest.mark.asyncio
async def test_detail_outcome_plan_projects_every_item_state(client, db_session, test_project):
    url, body = await _create_detail(client, test_project)
    run = await db_session.get(OrchestrationRun, uuid.UUID(body["run"]["id"]))
    author, adopted_task = uuid.uuid4(), uuid.uuid4()
    planning_task = Task(project_id=test_project.id, title="plan", status="done",
                         metadata_={"orchestration_contract": {"scope": "Plan the work"}})
    db_session.add(planning_task)
    await db_session.flush()

    # (item id, task status or None for unreleased, gate status): one row per plan_item_state outcome.
    seeds = [("queued", None, None), ("backlog", "backlog", "open"), ("started", "in_progress", "open"),
             ("pending", "done", "open"), ("accepted", "done", "accepted"), ("failed", "failed", "failed"),
             ("cancelled", "cancelled", "open")]
    expanded = []
    for item_id, task_status, gate_status in seeds:
        if task_status is None:
            continue
        task = Task(project_id=test_project.id, title=item_id, status=task_status)
        gate = OrchestrationGate(run_id=run.id, success_criterion_key=item_id, gate_type="work_completed",
                                 required_evidence={}, status=gate_status)
        db_session.add_all([task, gate])
        await db_session.flush()
        expanded.append({"plan_item_id": item_id, "task_id": str(task.id), "gate_id": str(gate.id),
                         **({"adopted": True} if item_id == "started" else {})})
    snapshot_items = [{"id": item_id, "work_function": "implementation", "scope": "s", "deliverable": "d"}
                      for item_id, _, _ in seeds]
    snapshot_items[1]["depends_on"] = ["plan_item:queued"]
    snapshot_items[2].update({"priority": 80, "why_runnable": "ready now", "adopt_task_id": str(adopted_task),
                              "artifact_ids": [str(adopted_task)], "title": "Started item"})
    artifact = Artifact(
        project_id=test_project.id, name="plan", artifact_type="plan", linked_task_id=planning_task.id,
        metadata_={"plan_kind": "implementation", "summary": "The plan", "planning_version": 2,
                   "plan_author_agent_id": str(author), "plan_items": []},
    )
    db_session.add(artifact)
    await db_session.flush()
    run.plan_state = {"status": "accepted", "accepted_artifact_id": str(artifact.id),
                      "accepted_plan_snapshot": {"items": snapshot_items}, "expanded_items": expanded}
    await db_session.flush()

    plan = (await client.get(url)).json()["plan"]

    assert set(plan) == PLAN_KEYS
    assert (plan["status"], plan["kind"], plan["author_agent_id"], plan["planning_version"], plan["summary"]) == (
        "accepted", "implementation", str(author), 2, "The plan")
    assert (plan["artifact_id"], plan["planning_task_id"], plan["scope"]) == (
        str(artifact.id), str(planning_task.id), "Plan the work")
    items = {i["key"]: i for i in plan["items"]}
    assert [i["key"] for i in plan["items"]] == [s[0] for s in seeds]
    assert all(set(i) == PLAN_ITEM_KEYS for i in items.values())
    assert {k: i["state"] for k, i in items.items()} == {
        "queued": "queued", "backlog": "queued", "started": "started", "pending": "evidence_pending",
        "accepted": "accepted", "failed": "failed", "cancelled": "cancelled"}
    assert items["backlog"]["depends_on"] == ["queued"]
    started = items["started"]
    assert (started["priority"], started["why_runnable"], started["adopt_task_id"], started["adopted"],
            started["artifact_ids"], started["title"]) == (
        80, "ready now", str(adopted_task), True, [str(adopted_task)], "Started item")
    assert items["queued"]["task_id"] is None and items["queued"]["priority"] == 50
    assert items["queued"]["why_runnable"] == "" and items["queued"]["adopted"] is False
    assert items["accepted"]["gate_id"] and items["accepted"]["task_id"]


@pytest.mark.asyncio
async def test_detail_plan_is_lenient_and_absent_without_artifact(client, db_session, test_project):
    url, body = await _create_detail(client, test_project)
    assert (await client.get(url)).json()["plan"] is None

    run = await db_session.get(OrchestrationRun, uuid.UUID(body["run"]["id"]))
    artifact = Artifact(project_id=test_project.id, name="plan", artifact_type="plan",
                        metadata_={"plan_kind": "bogus", "plan_items": [
                            {"id": "a", "priority": "high", "depends_on": "nope"}, "junk", {"title": "no key"}]})
    db_session.add(artifact)
    await db_session.flush()
    # Never accepted and malformed state: the 409-raising accepted-plan path is not used.
    run.plan_state = {"status": "proposed", "plan_artifact_id": str(artifact.id), "planning_version": "x"}
    await db_session.flush()

    plan = (await client.get(url)).json()["plan"]
    assert (plan["status"], plan["kind"], plan["planning_version"], plan["author_agent_id"]) == (
        "proposed", None, None, None)
    assert [(i["key"], i["state"], i["priority"], i["depends_on"]) for i in plan["items"]] == [
        ("a", "queued", 50, [])]


@pytest.mark.asyncio
async def test_detail_roadmap_plan_joins_released_items_and_hides_private_snapshot(client, db_session, test_project):
    from tests.test_orchestration_roadmap_children import goal_item, release

    _, parent, _, _, _ = await release(db_session, test_project, goal_item("alpha"))
    detail = (await client.get(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{parent.id}")).json()
    plan = detail["plan"]

    assert set(plan) == PLAN_KEYS and plan["artifact_id"]
    assert len(plan["items"]) == 1
    item = plan["items"][0]
    assert set(item) == PLAN_ITEM_KEYS
    assert (item["key"], item["unit_type"], item["state"]) == ("alpha", "goal", "started")
    assert item["child_goal_id"] == detail["children"][0] and item["gate_id"] and item["task_id"] is None
    dumped = json.dumps(plan)
    assert "parent_contract_snapshot" not in dumped and "workspace_policy" not in dumped
    assert "allocation" not in dumped and "objective" not in dumped


@pytest.mark.asyncio
async def test_detail_roadmap_plan_unreleased_item_is_queued(client, db_session, test_project):
    from tests.test_orchestration_roadmap_children import accepted_goal_roadmap, goal_item

    _, parent, _, _ = await accepted_goal_roadmap(db_session, test_project, goal_item("beta"))
    plan = (await client.get(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{parent.id}")).json()["plan"]

    assert [(i["key"], i["state"], i["child_goal_id"], i["gate_id"]) for i in plan["items"]] == [
        ("beta", "queued", None, None)]


def _report(status="done", next_step="verify"):
    return {"status": status, "changes": ["c"], "evidence": ["e"], "criterion_progress": {"done": "claimed"},
            "decisions": [], "risks": [], "open_questions": [], "next_step": next_step, "collaboration_need": None}


@pytest.mark.asyncio
async def test_detail_task_reports_project_only_validated_payloads(client, db_session, test_project):
    url, body = await _create_detail(client, test_project)
    run_id = uuid.UUID(body["run"]["id"])
    valid_task, bad_task, planning_task = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    valid_session = uuid.uuid4()
    for task_id, request in (
        (valid_task, {"task_id": str(valid_task), "session_id": str(valid_session), "report": _report()}),
        (bad_task, {"task_id": str(bad_task), "session_id": None, "report": {"status": "done"}}),
        (planning_task, {"task_id": str(planning_task), "session_id": str(uuid.uuid4())}),
    ):
        db_session.add(OrchestrationAction(
            run_id=run_id, idempotency_key=f"run:{run_id}:kind:report_consumed:task:{task_id}",
            action_type="report_consumed", request=request, target_type="task", target_id=task_id,
            status="completed",
        ))
    await db_session.flush()

    reports = (await client.get(url)).json()["task_reports"]

    by_task = {r["task_id"]: r for r in reports}
    assert set(by_task) == {str(valid_task), str(bad_task)}  # planning marker has no report key
    assert set(by_task[str(valid_task)]) == {"task_id", "session_id", "summary", "report"}
    assert by_task[str(valid_task)]["summary"] == "done · next: verify"
    assert by_task[str(valid_task)]["session_id"] == str(valid_session)
    assert by_task[str(valid_task)]["report"]["next_step"] == "verify"
    assert by_task[str(bad_task)]["report"] is None and by_task[str(bad_task)]["summary"] is None


@pytest.mark.asyncio
async def test_detail_task_reports_never_expose_raw_session_output(client, db_session, test_project):
    from tests.test_orchestration_runtime_e2e import _agent

    url, body = await _create_detail(client, test_project)
    run_id = uuid.UUID(body["run"]["id"])
    agent = _agent("reporter", ["implementation"])
    db_session.add(agent)
    await db_session.flush()
    task = Task(project_id=test_project.id, title="t", status="done", assigned_to=agent.id)
    db_session.add(task)
    await db_session.flush()
    secret = "RAW-SESSION-OUTPUT-SECRET"
    session = Session(project_id=test_project.id, task_id=task.id, agent_id=agent.id, adapter_type="claude_code", status="completed",
                      output=json.dumps({**_report(), "scratch": secret}))
    db_session.add(session)
    await db_session.flush()
    db_session.add(OrchestrationAction(
        run_id=run_id, idempotency_key=f"run:{run_id}:kind:report_consumed:task:{task.id}",
        action_type="report_consumed", target_type="task", target_id=task.id, status="completed",
        request={"task_id": str(task.id), "session_id": str(session.id), "report": None,
                 "report_validation_errors": ["report contains unexpected fields: scratch"]},
    ))
    await db_session.flush()

    response = await client.get(url)

    assert secret not in response.text
    report = response.json()["task_reports"][0]
    assert report["report"] is None and report["session_id"] == str(session.id)


@pytest.mark.asyncio
async def test_detail_exposes_authorization_wait_for_stale_start_queued_task(client, db_session, test_project):
    from tests.test_orchestration_authorization import _started_with_backlog_task

    svc, goal, run, task = await _started_with_backlog_task(db_session, test_project)
    url = f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}"
    assert (await client.get(url)).json()["authorization_wait"] is None

    goal.objective = "Changed after Start"
    await db_session.flush()
    assert await svc._start_released_tasks(db_session, goal, run) == 0

    wait = (await client.get(url)).json()["authorization_wait"]
    assert wait == {"task_ids": [str(task.id)], "reason_code": "contract_changed_since_start",
                    "required_decision": "start"}

    await svc.start_run(db_session, test_project.id, goal.id, actor="human:a")  # re-Start
    assert await svc._start_released_tasks(db_session, goal, run) == 1
    assert (await client.get(url)).json()["authorization_wait"] is None


@pytest.mark.asyncio
async def test_detail_exposes_capacity_wait_for_startable_queued_task(client, db_session, test_project):
    from tests.test_orchestration_progress_view import _run_with_criteria, _task
    from tests.test_orchestration_runtime_e2e import _agent

    goal, run = await _run_with_criteria(db_session, test_project, [])
    agent = _agent("detail-capacity", ["implementation"])
    db_session.add(agent)
    await db_session.flush()
    queued = await _task(db_session, test_project, run, status="backlog")
    queued.assigned_to = agent.id
    await db_session.flush()
    url = f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}"
    assert (await client.get(url)).json()["capacity_wait"] is None
    await _task(db_session, test_project, run, status="in_progress")
    await _task(db_session, test_project, run, status="ready")

    response = await client.get(url)

    assert response.status_code == 200
    assert response.json()["capacity_wait"] == {
        "root_goal_id": str(goal.id), "available_slots": 0, "queued_task_ids": [str(queued.id)],
    }


@pytest.mark.asyncio
async def test_detail_proposed_roadmap_plan_shows_item_key_items_queued(client, db_session, test_project):
    url, body = await _create_detail(client, test_project)
    run = await db_session.get(OrchestrationRun, uuid.UUID(body["run"]["id"]))
    artifact = Artifact(project_id=test_project.id, name="plan", artifact_type="plan",
                        metadata_={"plan_kind": "roadmap", "plan_items": [{"item_key": "alpha", "title": "A"}]})
    db_session.add(artifact)
    await db_session.flush()
    run.plan_state = {"status": "proposed", "plan_artifact_id": str(artifact.id)}
    await db_session.flush()

    plan = (await client.get(url)).json()["plan"]
    assert [(i["key"], i["state"]) for i in plan["items"]] == [("alpha", "queued")]


@pytest.mark.asyncio
async def test_detail_ignores_malformed_authorization_wait(client, db_session, test_project):
    from tests.test_orchestration_authorization import _started_with_backlog_task

    _, goal, _, task = await _started_with_backlog_task(db_session, test_project)
    url = f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}"
    for bad in ("x", {}):
        orch = {**(task.metadata_ or {}).get("orchestration", {}), "authorization_wait": bad}  # keep run_id
        task.metadata_ = {**(task.metadata_ or {}), "orchestration": orch}
        flag_modified(task, "metadata_")
        await db_session.flush()
        response = await client.get(url)
        assert response.status_code == 200
        assert response.json()["authorization_wait"] is None


@pytest.mark.asyncio
async def test_ordinary_decisions_stay_backward_compatible_with_null_recovery(
    client, db_session, auth_headers, test_project,
):
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService

    goal = OrchestrationGoal(
        project_id=test_project.id, objective="Ordinary decision goal",
        success_criteria=[], constraints={}, budget={},
    )
    db_session.add(goal)
    await db_session.flush()
    decision = await OrchestrationAuthorityDecisionService().create_pending(
        db_session, goal.id, decision_key="ordinary:choice", title="Pick", question="Which?",
        authority="human", options=[{"key": "a", "label": "A"}, {"key": "b", "label": "B"}],
    )
    base = f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/decisions"

    listed = await client.get(base, headers=auth_headers)
    assert listed.status_code == 200
    (item,) = listed.json()
    assert "recovery" in item and item["recovery"] is None
    checkpoint = await client.get(f"{base}/checkpoint", headers=auth_headers)
    assert checkpoint.json()["items"][0]["recovery"] is None

    answered = await client.post(f"{base}/{decision.id}/answer", json={"selected_option": "a"}, headers=auth_headers)
    assert answered.status_code == 200
    assert answered.json()["decision"]["status"] == "answered"
    assert answered.json()["decision"]["recovery"] is None
    assert answered.json()["continuation_applied"] is False


@pytest.mark.asyncio
async def test_goal_detail_projects_current_unfulfilled_meeting_commitments(client, db_session, test_project):
    from tests.test_orchestration_delegation_contracts import _make_run
    from tests.test_orchestration_progress_view import _task, _meeting, _action_item
    _, goal, run = await _make_run(db_session, test_project.id)
    source = await _task(db_session, test_project, run)
    meeting = await _meeting(db_session, test_project, source)
    live = await _task(db_session, test_project, run, status="in_progress")
    tracked = await _action_item(db_session, meeting, description="Live obligation", task_id=live.id, status="task_created")
    open_item = await _action_item(db_session, meeting, description="Open obligation")
    await _action_item(db_session, meeting, description="Waived", status="dismissed")
    done = await _task(db_session, test_project, run, status="done")
    await _action_item(db_session, meeting, description="Fulfilled", task_id=done.id, status="task_created")
    # A historical run of this same goal must not bleed into the current projection.
    from datetime import timedelta
    historical = OrchestrationRun(goal_id=goal.id, status="cancelled", phase="completed", started_at=run.started_at - timedelta(days=1))
    db_session.add(historical)
    await db_session.flush()
    old_source = await _task(db_session, test_project, historical)
    await _action_item(db_session, await _meeting(db_session, test_project, old_source), description="Old obligation")
    response = await client.get(f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}")
    assert response.status_code == 200, response.text
    commitments = {entry["id"]: entry for entry in response.json()["meeting_commitments"]}
    assert set(commitments) == {str(tracked.id), str(open_item.id)}
    assert commitments[str(tracked.id)] == {
        "id": str(tracked.id), "meeting_id": str(meeting.id), "meeting_title": meeting.title,
        "description": "Live obligation", "state": "task_in_flight", "task_id": str(live.id),
        "graph_run_id": None, "reason": "linked task still in flight",
    }
    assert commitments[str(open_item.id)]["state"] == "open"
    assert commitments[str(open_item.id)]["task_id"] is None


@pytest.mark.asyncio
async def test_resume_persists_one_goal_resumed_event_and_repeat_resume_adds_none(client, db_session, test_project):
    from huddleroom.models.event_log import EventLog
    from huddleroom.services.orchestration_service import GOAL_RESUMED_EVENT_TYPE

    pid = test_project.id
    goal_id = await _create_goal_id(client, pid)
    assert (await client.post(f"/api/v1/projects/{pid}/orchestration/goals/{goal_id}/pause")).status_code == 200
    assert (await client.post(f"/api/v1/projects/{pid}/orchestration/goals/{goal_id}/resume")).status_code == 200
    assert (await client.post(f"/api/v1/projects/{pid}/orchestration/goals/{goal_id}/resume")).status_code == 409

    rows = (await db_session.scalars(
        select(EventLog).where(EventLog.event_type == GOAL_RESUMED_EVENT_TYPE)
    )).all()
    assert len(rows) == 1
    assert rows[0].payload["goal_id"] == str(goal_id)
