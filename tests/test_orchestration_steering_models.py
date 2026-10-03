import os
import sqlite3
import subprocess
import sys
import uuid

import pytest
from sqlalchemy.exc import IntegrityError


STEERING_TABLES = {
    "orchestration_steering_state",
    "orchestration_steering_proposals",
    "orchestration_steering_requests",
    "orchestration_steering_transitions",
    "orchestration_steering_result_links",
}


def _alembic(env, *command):
    result = subprocess.run(
        [sys.executable, "-m", "alembic", *command],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr


def _env(tmp_path):
    return {**os.environ, "RALLY_DATABASE_URL": f"sqlite+aiosqlite:///{tmp_path / 'steering.db'}"}


def _tables(tmp_path):
    with sqlite3.connect(tmp_path / "steering.db") as connection:
        return {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def test_043_round_trip_and_default_on(tmp_path, monkeypatch):
    from huddleroom.config import Settings

    env = _env(tmp_path)
    _alembic(env, "upgrade", "042")
    _alembic(env, "upgrade", "043")
    assert _tables(tmp_path) >= STEERING_TABLES
    monkeypatch.delenv("RALLY_ORCHESTRATION_CONVERSATION_STEERING_ENABLED", raising=False)
    assert Settings(_env_file=None).orchestration_conversation_steering_enabled is True
    monkeypatch.setenv("RALLY_ORCHESTRATION_CONVERSATION_STEERING_ENABLED", "false")
    assert Settings(_env_file=None).orchestration_conversation_steering_enabled is False
    _alembic(env, "downgrade", "042")
    assert not _tables(tmp_path) & STEERING_TABLES


def test_043_schema_has_exact_columns_indexes_and_checks(tmp_path):
    env = _env(tmp_path)
    _alembic(env, "upgrade", "043")
    expected_columns = {
        "orchestration_steering_state": {"id", "goal_id", "inbox_version", "direction_version", "created_at", "updated_at"},
        "orchestration_steering_proposals": {"id", "response_id", "goal_id", "actor_id", "status", "draft", "dismissed_at", "promoted_request_id", "created_at", "updated_at"},
        "orchestration_steering_requests": {"id", "goal_id", "actor_id", "client_request_id", "sequence", "submitted_run_id", "directive", "target_type", "target_id", "scope", "lifetime", "impact_summary", "source_proposal_id", "supersedes_request_id", "status", "reason_code", "contract_version", "plan_version", "submitted_at", "considered_at", "finished_at", "updated_at"},
        "orchestration_steering_transitions": {"id", "request_id", "sequence", "from_status", "to_status", "reason_code", "actor", "created_at"},
        "orchestration_steering_result_links": {"id", "request_id", "decision_id", "action_id", "created_at"},
    }
    with sqlite3.connect(tmp_path / "steering.db") as connection:
        for table, columns in expected_columns.items():
            assert {row[1] for row in connection.execute(f"PRAGMA table_info({table})")} == columns
            sql = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()[0]
            for name in {
                "orchestration_steering_state": ("uq_orch_steering_state_goal", "ck_orch_steering_state_versions"),
                "orchestration_steering_proposals": ("uq_orch_steering_proposals_response", "ck_orch_steering_proposals_status", "ck_orch_steering_proposals_lifecycle"),
                "orchestration_steering_requests": ("uq_orch_steering_requests_goal_actor_client", "uq_orch_steering_requests_goal_sequence", "ck_orch_steering_requests_status", "ck_orch_steering_requests_target_type", "ck_orch_steering_requests_directive_length", "ck_orch_steering_requests_scope_lifetime"),
                "orchestration_steering_transitions": ("uq_orch_steering_transitions_request_sequence", "ck_orch_steering_transitions_sequence", "ck_orch_steering_transitions_status"),
                "orchestration_steering_result_links": ("uq_orch_steering_result_links_request_decision_action",),
            }[table]:
                assert name in sql
        indexes = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='index'")}
        assert {"idx_orch_steering_requests_goal_status_sequence", "idx_orch_steering_proposals_goal_status", "idx_orch_steering_transitions_request_sequence"} <= indexes


def test_steering_identities_are_stable():
    from huddleroom.models.orchestration_steering import (
        steering_proposal_id, steering_request_id, steering_result_link_id, steering_state_id,
        steering_transition_id,
    )

    goal_id = uuid.UUID("11111111-1111-1111-1111-111111111111")
    actor_id = uuid.UUID("22222222-2222-2222-2222-222222222222")
    client_request_id = uuid.UUID("33333333-3333-3333-3333-333333333333")
    response_id = uuid.UUID("44444444-4444-4444-8444-444444444444")
    decision_id = uuid.UUID("55555555-5555-5555-8555-555555555555")
    action_id = uuid.UUID("66666666-6666-6666-8666-666666666666")
    request_id = steering_request_id(goal_id, actor_id, client_request_id)
    assert steering_state_id(goal_id) == steering_state_id(goal_id)
    assert steering_proposal_id(response_id) == steering_proposal_id(response_id)
    assert request_id == steering_request_id(goal_id, actor_id, client_request_id)
    assert steering_transition_id(request_id, 1) == steering_transition_id(request_id, 1)
    assert steering_result_link_id(request_id, decision_id, action_id) == steering_result_link_id(request_id, decision_id, action_id)
    assert request_id != steering_request_id(goal_id, actor_id, uuid.uuid4())


@pytest.mark.asyncio
async def test_steering_constraints_reject_invalid_rows(db_session, conversation_goal_run, test_user):
    from huddleroom.models import OrchestrationSteeringProposal, OrchestrationSteeringRequest, OrchestrationSteeringState, OrchestrationSteeringTransition
    from huddleroom.models.orchestration_conversation import ConversationMessage, ConversationResponse

    goal, _ = conversation_goal_run
    message = ConversationMessage(goal_id=goal.id, actor_id=test_user.id, client_request_id=uuid.uuid4(), sequence=1, content="question")
    db_session.add(message)
    await db_session.flush()
    response = ConversationResponse(message_id=message.id, dossier={}, context_manifest={}, context_version="v1", provider_request_id="rally-chat:steering-constraints")
    db_session.add(response)
    request = OrchestrationSteeringRequest(id=uuid.uuid4(), goal_id=goal.id, actor_id=test_user.id, client_request_id=uuid.uuid4(), sequence=1, directive="x", target_type="goal", target_id=str(goal.id), scope="goal", lifetime="future_runs", impact_summary="x", status="pending", reason_code="submitted", contract_version="none", plan_version="none")
    db_session.add(request)
    await db_session.flush()

    records = (
        OrchestrationSteeringState(id=uuid.uuid4(), goal_id=goal.id, inbox_version=-1, direction_version=0),
        OrchestrationSteeringRequest(id=uuid.uuid4(), goal_id=goal.id, actor_id=test_user.id, client_request_id=uuid.uuid4(), sequence=2, directive="x", target_type="goal", target_id=str(goal.id), scope="goal", lifetime="remaining_current_run", impact_summary="x", status="pending", reason_code="submitted", contract_version="none", plan_version="none"),
        OrchestrationSteeringRequest(id=uuid.uuid4(), goal_id=goal.id, actor_id=test_user.id, client_request_id=uuid.uuid4(), sequence=3, directive="x", target_type="unknown", target_id=str(goal.id), scope="goal", lifetime="future_runs", impact_summary="x", status="pending", reason_code="submitted", contract_version="none", plan_version="none"),
        OrchestrationSteeringRequest(id=uuid.uuid4(), goal_id=goal.id, actor_id=test_user.id, client_request_id=uuid.uuid4(), sequence=4, directive="x", target_type="goal", target_id=str(goal.id), scope="goal", lifetime="future_runs", impact_summary="x", status="unknown", reason_code="submitted", contract_version="none", plan_version="none"),
        OrchestrationSteeringRequest(id=uuid.uuid4(), goal_id=goal.id, actor_id=test_user.id, client_request_id=uuid.uuid4(), sequence=5, directive="", target_type="goal", target_id=str(goal.id), scope="goal", lifetime="future_runs", impact_summary="x", status="pending", reason_code="submitted", contract_version="none", plan_version="none"),
        OrchestrationSteeringRequest(id=uuid.uuid4(), goal_id=goal.id, actor_id=test_user.id, client_request_id=uuid.uuid4(), sequence=6, directive="x" * 4001, target_type="goal", target_id=str(goal.id), scope="goal", lifetime="future_runs", impact_summary="x", status="pending", reason_code="submitted", contract_version="none", plan_version="none"),
        OrchestrationSteeringTransition(id=uuid.uuid4(), request_id=request.id, sequence=0, to_status="pending", reason_code="submitted", actor="operator"),
        OrchestrationSteeringTransition(id=uuid.uuid4(), request_id=request.id, sequence=1, to_status="unknown", reason_code="submitted", actor="operator"),
        OrchestrationSteeringProposal(id=uuid.uuid4(), response_id=response.id, goal_id=goal.id, actor_id=test_user.id, status="dismissed", draft={}, dismissed_at=None),
        OrchestrationSteeringProposal(id=uuid.uuid4(), response_id=response.id, goal_id=goal.id, actor_id=test_user.id, status="unknown", draft={}, dismissed_at=None),
    )
    for record in records:
        with pytest.raises(IntegrityError):
            async with db_session.begin_nested():
                db_session.add(record)
                await db_session.flush()
