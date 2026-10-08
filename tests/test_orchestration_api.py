import uuid

import pytest
from sqlalchemy import func, select

from huddleroom.models.orchestration import (
    OrchestrationAction,
    OrchestrationEvidence,
    OrchestrationGate,
    OrchestrationRun,
)
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
