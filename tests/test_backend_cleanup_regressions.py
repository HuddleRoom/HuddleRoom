"""Regression coverage for recovery identities and caller provenance."""
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from huddleroom.models.artifact import Artifact
from huddleroom.models.orchestration import OrchestrationAction
from huddleroom.models.session import Session
from huddleroom.services.orchestration_recovery_service import OrchestrationRecoveryService, RunnerObservation
from huddleroom.services.orchestration_service import OrchestrationService
from tests.test_orchestration_plan_to_work_expansion import (
    _adoption_setup, _adopting_item, _assert_adoption_rejected, _make_run,
)
from tests.test_orchestration_worker_ownership_recovery import _owned


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_provenance", ["other_type", "other_run"])
async def test_plan_adoption_requires_delegation_action_of_same_run(db_session, test_project, invalid_provenance):
    service, _, run, planner, _, request_action, task = await _adoption_setup(db_session, test_project.id)
    action = await db_session.get(OrchestrationAction, task.id)
    assert action is not None
    if invalid_provenance == "other_type":
        action.action_type = "request_plan"
    else:
        _, _, other_run = await _make_run(db_session, test_project.id)
        action.run_id = other_run.id
    await db_session.flush()

    await _assert_adoption_rejected(
        db_session, test_project.id, service, run, planner, request_action,
        [_adopting_item(task.id)], "was not created by a delegation",
    )


@pytest.mark.asyncio
async def test_artifact_create_ignores_caller_supplied_agent(client, auth_headers, test_project, test_agent, db_session):
    response = await client.post(
        f"/api/v1/projects/{test_project.id}/artifacts",
        headers=auth_headers,
        json={"name": "Spoofed author", "artifact_type": "plan", "created_by_agent": str(test_agent.id)},
    )
    assert response.status_code == 201
    artifact = await db_session.get(Artifact, UUID(response.json()["id"]))
    assert artifact.created_by_agent is None


@pytest.mark.asyncio
async def test_repeated_pre_effect_interruptions_reserve_distinct_retry_actions(db_session, test_project, test_agent):
    goal, run, _, task, session = await _owned(
        db_session, test_project, test_agent, session_status="pending", approved_attempt=True,
        attempt={"claimed_runner_task_id": "runner-1", "attempt_version": 1,
                 "effect_state": "not_started", "usage_complete": False},
    )
    service = OrchestrationRecoveryService()
    first_snapshot = await service.build_goal_snapshot(db_session, goal.id, datetime.now(timezone.utc))
    first = await service.apply_goal_recovery(db_session, first_snapshot, {
        session.id: RunnerObservation(session.runner_task_id, "terminal_failure", datetime.now(timezone.utc)),
    })
    first_action = await db_session.get(OrchestrationAction, first.action_id)
    replay = await OrchestrationService().execute_retry_task_action(
        db_session, run.id, first_action.request, first_action.idempotency_key,
        recovery_disposition="interrupted_safe", source_session_id=session.id,
    )
    assert replay.id == first_action.id
    replacement = await db_session.get(Session, first_action.target_id)
    approved_owner = OrchestrationAction(
        run_id=run.id, idempotency_key=f"recovery:{uuid4()}:retry:1:effect", action_type="retry_task",
        status="completed", request={"task_id": str(task.id)}, target_type="session", target_id=replacement.id,
        dispatch_contract={"owner": "orchestration_recovery"},
    )
    db_session.add(approved_owner)
    await db_session.flush()
    task.metadata_ = {**task.metadata_, "orchestration": {
        **task.metadata_["orchestration"], "action_id": str(approved_owner.id),
    }}
    replacement.runner_task_id = "runner-2"
    replacement.metadata_ = {**replacement.metadata_, "orchestration": {"action_id": str(approved_owner.id)}, "attempt": {
        "claimed_runner_task_id": "runner-2", "attempt_version": 2,
        "effect_state": "not_started", "usage_complete": False,
    }}
    replacement.status = "pending"
    task.status = "in_progress"
    await db_session.flush()
    second_snapshot = await service.build_goal_snapshot(db_session, goal.id, datetime.now(timezone.utc))
    second = await service.apply_goal_recovery(db_session, second_snapshot, {
        replacement.id: RunnerObservation("runner-2", "terminal_failure", datetime.now(timezone.utc)),
    })
    second_action = await db_session.get(OrchestrationAction, second.action_id)
    assert first.classification == second.classification == "interrupted_safe"
    assert first_action.action_type == second_action.action_type == "retry_task"
    assert first_action.id != second_action.id
    assert first_action.idempotency_key != second_action.idempotency_key
    assert first_action.target_id != second_action.target_id
