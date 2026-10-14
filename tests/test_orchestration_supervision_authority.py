import uuid
from datetime import timedelta

import pytest
from fastapi import HTTPException
from sqlalchemy import func, select

from huddleroom.models.base import _utcnow
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGoal, OrchestrationRun, OrchestrationWait
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
from huddleroom.services.orchestration_service import OrchestrationService


pytestmark = pytest.mark.asyncio


async def _runtime_decision(db_session, test_project):
    goal = OrchestrationGoal(
        project_id=test_project.id,
        objective="Approve runtime continuation",
        success_criteria=[{"key": "done", "description": "Done"}],
        status="active",
    )
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id, status="running")
    db_session.add(run)
    await db_session.flush()
    service = OrchestrationAuthorityDecisionService()
    decision = await service.create_runtime_question(
        db_session, goal.id, run_id=run.id, subject="release:deployment",
        authority="human", contract_version="plan:1",
        continuation={"action_type": "continue", "reason": "approved deployment"},
        question="Approve deployment?", options=[{"key": "approve"}, {"key": "reject"}],
    )
    return service, goal, run, decision


async def _text_runtime_decision(db, project):
    service, goal, run, _ = await _runtime_decision(db, project)
    action = await OrchestrationService().execute_ask_human_action(
        db, run.id, {"action_type": "ask_human", "question": "What counts as success?",
                     "subject": "owner:success", "contract_version": "plan:1",
                     "continuation": {"action_type": "continue"}},
        f"run:{run.id}:kind:ask_human:owner-success",
    )
    return service, goal, run, await db.get(OrchestrationAuthorityDecision, action.target_id)


@pytest.mark.parametrize("reason", [None, "", " \n\u0085 ", "x" * 4001, "\uFEFF" + "x" * 4000],
                         ids=["missing", "empty", "blank", "4001", "feff-4001"])
async def test_runtime_text_answer_rejects_blank_and_over_limit_without_mutation(
    db_session, test_project, test_user, reason
):
    service, _, _, decision = await _text_runtime_decision(db_session, test_project)
    with pytest.raises(ValueError):
        await service.answer_runtime_question(db_session, decision, "acknowledge", reason=reason,
                                             actor_user_id=test_user.id, contract_version="plan:1")
    await db_session.refresh(decision)
    assert (decision.status, decision.selected_option, decision.reason, decision.decided_at) == (
        "pending", None, None, None)


@pytest.mark.parametrize("reason", ["  optional rationale \n", None, "x" * 4000, "x" * 4001],
                         ids=["normalized", "missing", "4000", "4001"])
async def test_runtime_meaningful_choice_reason_is_optional_and_bounded(
    db_session, test_project, test_user, reason
):
    service, _, _, decision = await _runtime_decision(db_session, test_project)
    if reason and len(reason.strip()) > 4000:
        with pytest.raises(ValueError):
            await service.answer_runtime_question(db_session, decision, "approve", reason=reason,
                                                 actor_user_id=test_user.id, contract_version="plan:1")
        assert decision.status == "pending"
    else:
        result = await service.answer_runtime_question(db_session, decision, "approve", reason=reason,
                                                      actor_user_id=test_user.id, contract_version="plan:1")
        assert result.decision.reason == (reason.strip() if reason is not None else None)


async def test_runtime_text_stale_contract_and_invalid_choice_are_inert(db_session, test_project, test_user):
    service, _, _, decision = await _text_runtime_decision(db_session, test_project)
    await service.answer_runtime_question(db_session, decision, "acknowledge", reason="stale text",
                                         actor_user_id=test_user.id, contract_version="old")
    with pytest.raises(ValueError):
        await service.answer_runtime_question(db_session, decision, "invented", reason="valid text",
                                             actor_user_id=test_user.id, contract_version="plan:1")
    await db_session.refresh(decision)
    assert (decision.status, decision.selected_option, decision.reason) == ("pending", None, None)


async def test_runtime_question_replays_pending_and_answered_identity(db_session, test_project, test_user):
    service, goal, run, first = await _runtime_decision(db_session, test_project)
    replay = await service.create_runtime_question(
        db_session, goal.id, run_id=run.id, subject="release:deployment", authority="human",
        contract_version="plan:1", continuation={"action_type": "continue"},
        question="Changed text must not replace durable question", options=[{"key": "approve"}],
    )
    assert replay.id == first.id
    await service.answer_runtime_question(
        db_session, first, "approve", actor_user_id=test_user.id, contract_version="plan:1"
    )
    answered_replay = await service.create_runtime_question(
        db_session, goal.id, run_id=run.id, subject="release:deployment", authority="human",
        contract_version="plan:1", continuation={"action_type": "continue"}, question="Again?",
        options=[{"key": "approve"}],
    )
    assert answered_replay.id == first.id
    assert await db_session.scalar(
        select(func.count(OrchestrationAuthorityDecision.id)).where(
            OrchestrationAuthorityDecision.goal_id == goal.id
        )
    ) == 1


async def test_ask_human_action_creates_a_durable_runtime_decision(db_session, test_project):
    goal = OrchestrationGoal(
        project_id=test_project.id, objective="Runtime ask", status="active",
        success_criteria=[{"key": "done", "description": "Done"}],
    )
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id, status="running")
    db_session.add(run)
    await db_session.flush()
    action = await OrchestrationService().execute_ask_human_action(
        db_session, run.id,
        {"action_type": "ask_human", "question": "Continue?", "subject": "task:1",
         "contract_version": "plan:1", "options": [{"key": "approve"}],
         "continuation": {"action_type": "continue"}},
        f"run:{run.id}:kind:ask_human:test",
    )
    decision = await db_session.get(OrchestrationAuthorityDecision, action.target_id)
    assert action.target_type == "authority_decision"
    assert decision.runtime_identity is not None


async def test_runtime_answer_applies_exactly_once_and_rejects_conflict(db_session, test_project, test_user):
    service, _, run, decision = await _runtime_decision(db_session, test_project)
    actor = test_user.id
    first = await service.answer_runtime_question(
        db_session, decision, "approve", actor_user_id=actor, contract_version="plan:1"
    )
    duplicate = await service.answer_runtime_question(
        db_session, decision, "approve", actor_user_id=actor, contract_version="plan:1"
    )
    assert first.continuation_applied is True
    assert duplicate.continuation_applied is False
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id,
        OrchestrationAction.idempotency_key == f"run:{run.id}:kind:decision_continuation:decision:{decision.id}",
    )) == 1
    with pytest.raises(ValueError, match="conflicting"):
        await service.answer_runtime_question(
            db_session, decision, "reject", actor_user_id=actor, contract_version="plan:1"
        )


@pytest.mark.parametrize("question", ["Approve deployment?", "Approve " + "deployment " * 30 + "?"])
async def test_ask_human_decision_can_be_fetched_and_answered_with_contract_version(
    db_session, test_project, client, auth_headers, question
):
    goal = OrchestrationGoal(
        project_id=test_project.id, objective="Approve deployment", status="active",
        success_criteria=[{"key": "done", "description": "Done"}],
    )
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id, status="running")
    db_session.add(run)
    await db_session.flush()
    reason = "The production rollout needs operator approval."
    action = await OrchestrationService().execute_ask_human_action(
        db_session, run.id,
        {"action_type": "ask_human", "question": question, "reason": reason,
         "contract_version": "plan:1", "options": [{"key": "approve"}],
         "continuation": {"action_type": "continue"}},
        f"run:{run.id}:kind:ask_human:deployment",
    )
    base = f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/decisions"
    response = await client.get(base, headers=auth_headers)
    assert response.status_code == 200
    decision = next(row for row in response.json() if row["id"] == str(action.target_id))
    assert decision["runtime_identity"]
    assert decision["contract_version"] == "plan:1"
    assert decision["title"] == question[:255]
    assert "run:" not in decision["title"]
    assert decision["question"] == question
    assert decision["context"] == reason
    missing_version = await client.post(
        f"{base}/{decision['id']}/answer", headers=auth_headers,
        json={"selected_option": "approve"},
    )
    assert missing_version.status_code == 400
    assert missing_version.json()["detail"] == "contract_version is required for runtime decisions"
    stale_version = await client.post(
        f"{base}/{decision['id']}/answer", headers=auth_headers,
        json={"selected_option": "approve", "contract_version": "plan:stale"},
    )
    assert stale_version.status_code == 200
    assert stale_version.json()["continuation_applied"] is False
    assert stale_version.json()["decision"]["status"] == "pending"
    answer = await client.post(
        f"{base}/{decision['id']}/answer", headers=auth_headers,
        json={"selected_option": "approve", "contract_version": decision["contract_version"]},
    )
    assert answer.status_code == 200, answer.text
    assert answer.json()["continuation_applied"] is True
    assert answer.json()["decision"]["status"] == "answered"
    continuation = await db_session.get(
        OrchestrationAction, uuid.UUID(answer.json()["decision"]["continuation_action_id"])
    )
    assert continuation.run_id == run.id
    assert continuation.status == "completed"


async def test_runtime_answer_rejects_unauthorized_actor_and_stale_contract(db_session, test_project, test_user):
    service, _, run, decision = await _runtime_decision(db_session, test_project)
    with pytest.raises(ValueError, match="human"):
        await service.answer_runtime_question(
            db_session, decision, "approve", actor_user_id=None, contract_version="plan:1"
        )
    stale = await service.answer_runtime_question(
        db_session, decision, "approve", actor_user_id=test_user.id, contract_version="plan:stale"
    )
    assert stale.continuation_applied is False
    assert decision.status == "pending"


@pytest.mark.parametrize("control", ["pause", "cancel", "continuous_stop"])
async def test_runtime_answer_records_audit_but_control_preempts_continuation(
    db_session, test_project, test_user, control
):
    service, goal, run, decision = await _runtime_decision(db_session, test_project)
    if control == "pause":
        goal.status = run.status = "paused"
    elif control == "cancel":
        goal.status = run.status = "cancelled"
    else:
        goal.goal_type = "continuous"
        goal.planning_mode = "continuous_policy"
        goal.continuous_state = {"stopped_at": "2026-09-09T00:00:00+00:00"}
    result = await service.answer_runtime_question(
        db_session, decision, "approve", actor_user_id=test_user.id, contract_version="plan:1"
    )
    assert result.decision.status == "answered"
    assert result.continuation_applied is False
    assert result.decision.continuation_action_id is None


async def test_runtime_answer_only_clears_decision_linked_blockers(db_session, test_project, test_user):
    service, _, run, decision = await _runtime_decision(db_session, test_project)
    run.active_blockers = [
        {"kind": "human", "decision_id": str(decision.id)},
        {"kind": "unrelated", "decision_id": str(uuid.uuid4())},
    ]
    linked_wait = OrchestrationWait(
        run_id=run.id, wait_key="linked", owner={"decision_id": str(decision.id)},
        awaited_event={}, due_recheck_at=_utcnow() + timedelta(minutes=1), fallback={},
    )
    unrelated_wait = OrchestrationWait(
        run_id=run.id, wait_key="unrelated", owner={"decision_id": str(uuid.uuid4())},
        awaited_event={}, due_recheck_at=_utcnow() + timedelta(minutes=1), fallback={},
    )
    db_session.add_all([linked_wait, unrelated_wait])
    await db_session.flush()
    await service.answer_runtime_question(
        db_session, decision, "approve", actor_user_id=test_user.id, contract_version="plan:1"
    )
    assert run.active_blockers == [{"kind": "unrelated", "decision_id": run.active_blockers[0]["decision_id"]}]
    assert linked_wait.status == "cleared"
    assert unrelated_wait.status == "open"
