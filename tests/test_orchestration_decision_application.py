"""Shared application retains the complete specialized decision paths."""
import json
import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy import func, select

from huddleroom.models import User
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGoal
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
from huddleroom.schemas.orchestration import (
    OrchestrationAgentDefinitionReviewBatchAnswerRequest, OrchestrationDecisionAnswerRequest,
)
from huddleroom.services import orchestration_decision_application as application
from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
from tests.test_orchestration_supervision_authority import _runtime_decision
from tests.test_orchestration_supervision_authority import _text_runtime_decision
from huddleroom.models.event_log import EventLog
from tests.test_orchestration_recovery_authority import actions_of, diagnosed, fail_on_effect, goal_run
from tests.test_orchestration_agent_definition_review import _three_pending_proposals, review_goal, review_run
from tests.test_orchestration_roster_mutation import (
    PROPOSED_DESCRIPTION, agents_named, approve_semantic_reviews, assert_entry_handoff, current_process,
    decisions_for, fresh_agent, interfere_after_roster_release, lock_tracer, outcome_or_conflict,
    proposed_agent_definition, seed_definition_proposal, seed_hierarchy_proposal, sessions, single_agent_write,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("safe_goal_analysis")]


async def _answer(db, goal, decision, user, **data):
    return await application.answer_decision(db, goal.project_id, goal.id, decision.id,
                                            OrchestrationDecisionAnswerRequest(**data), user)


@pytest.mark.parametrize("text", ["a\n😀" * 1333 + "z", "\uFEFFhello\uFEFF", "a\n😀" * 1333 + "zz"],
                         ids=["4000-multiline-astral", "feff", "4001-multiline-astral"])
async def test_runtime_text_answer_persists_normalized_reason_and_bounds_codepoints(
    db_session, sessions, test_project, test_user, text
):
    from httpx import ASGITransport, AsyncClient
    from huddleroom.main import create_app
    from huddleroom.database import get_db
    from huddleroom.dependencies import get_current_user

    _, goal, _, decision = await _text_runtime_decision(db_session, test_project)
    decision_id = decision.id
    url = f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/decisions/{decision.id}/answer"
    await db_session.commit()
    app = create_app()

    async def request_db():
        async with sessions() as db:
            yield db
            await db.commit()

    async def actor():
        return test_user

    app.dependency_overrides[get_db] = request_db
    app.dependency_overrides[get_current_user] = actor
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        stale = await client.post(url, json={"selected_option": "acknowledge", "reason": "stale answer",
                                            "contract_version": "old"})
        assert stale.status_code == 200 and stale.json()["continuation_applied"] is False
        async with sessions() as fresh:
            saved = await fresh.get(OrchestrationAuthorityDecision, decision_id)
            assert (saved.status, saved.selected_option, saved.reason) == ("pending", None, None)
        answer = await client.post(url, json={
            "selected_option": "acknowledge", "reason": " \u0085" + text + " \n", "contract_version": "plan:1"})
    if len(text) > 4000:
        assert answer.status_code == 409, answer.text
        async with sessions() as fresh:
            saved = await fresh.get(OrchestrationAuthorityDecision, decision_id)
            assert (saved.status, saved.selected_option, saved.reason) == ("pending", None, None)
        return
    assert answer.status_code == 200, answer.text
    assert answer.json()["decision"]["reason"] == text
    async with sessions() as fresh:
        saved = await fresh.get(OrchestrationAuthorityDecision, decision_id)
        assert (saved.status, saved.selected_option, saved.reason) == ("answered", "acknowledge", text)


async def _seed_committed_text(sessions, project, user):
    async with sessions() as db:
        _, goal, run, decision = await _text_runtime_decision(db, project)
        ids = (goal.id, run.id, decision.id, user.id, project.id)
        await db.commit()
        return ids


async def _fresh_text_answer(db, ids, reason):
    goal_id, _, decision_id, user_id, project_id = ids
    return await application.answer_decision(db, project_id, goal_id, decision_id,
        OrchestrationDecisionAnswerRequest(selected_option="acknowledge", reason=reason, contract_version="plan:1"),
        await db.get(User, user_id))


async def test_runtime_answer_precommit_failure_rolls_back_in_fresh_session(
    db_session, sessions, test_project, test_user, monkeypatch
):
    await db_session.commit()
    ids = await _seed_committed_text(sessions, test_project, test_user)
    from huddleroom.services import orchestration_authority_service as authority
    original = authority.emit_event_once

    async def fail_after_answer(*args, **kwargs):
        if args[2] == "authority.decision_resolved":
            raise RuntimeError("precommit injected failure")
        return await original(*args, **kwargs)

    monkeypatch.setattr(authority, "emit_event_once", fail_after_answer)
    async with sessions() as db:
        with pytest.raises(RuntimeError, match="precommit injected"):
            await _fresh_text_answer(db, ids, "Keep the exact answer")
        await db.rollback()
    async with sessions() as fresh:
        saved = await fresh.get(OrchestrationAuthorityDecision, ids[2])
        assert (saved.status, saved.selected_option, saved.reason, saved.continuation_action_id) == (
            "pending", None, None, None)
        assert await fresh.scalar(select(func.count()).select_from(EventLog).where(
            EventLog.dedup_key == f"authority.decision_resolved:{ids[2]}")) == 0


@pytest.mark.parametrize("changed", [False, True])
async def test_runtime_answer_response_loss_replays_exact_payload_once(
    db_session, sessions, test_project, test_user, changed
):
    await db_session.commit()
    ids = await _seed_committed_text(sessions, test_project, test_user)
    async with sessions() as db:
        text = "a\n😀" * 1333 + "z"
        result = await _fresh_text_answer(db, ids, "  " + text + "  ")
        await db.commit()
        timestamp = result.decision.decided_at
        # The response is deliberately discarded after the real commit.
    async with sessions() as fresh:
        if changed:
            with pytest.raises(HTTPException) as error:
                await _fresh_text_answer(fresh, ids, "Changed answer")
            assert error.value.status_code == 409
            await fresh.rollback()
        else:
            replay = await _fresh_text_answer(fresh, ids, text)
            assert replay.continuation_applied is False
            await fresh.commit()
    async with sessions() as fresh:
        saved = await fresh.get(OrchestrationAuthorityDecision, ids[2])
        assert saved.reason == text
        assert saved.decided_at == timestamp
        assert await fresh.scalar(select(func.count()).select_from(EventLog).where(
            EventLog.dedup_key == f"authority.decision_resolved:{ids[2]}")) == 1
        actions = (await fresh.scalars(select(OrchestrationAction).where(
            OrchestrationAction.idempotency_key == f"run:{ids[1]}:kind:decision_continuation:decision:{ids[2]}"))).all()
        assert len(actions) == 1 and actions[0].status == "completed"
        assert saved.continuation_action_id == actions[0].id


async def test_ordinary_human_answer_preserves_options_and_audit(db_session, conversation_goal_run, test_user):
    goal, run = conversation_goal_run
    decision = await OrchestrationAuthorityDecisionService().create_pending(
        db_session, goal.id, decision_key="chat-test", title="Choose", question="Continue?",
        options=["approve", "reject"], authority="human", run_id=run.id,
    )
    with pytest.raises(HTTPException) as error:
        await _answer(db_session, goal, decision, test_user, selected_option="invented")
    assert error.value.status_code == 400 and decision.status == "pending"
    response = await _answer(db_session, goal, decision, test_user, selected_option="approve", reason="Proceed")
    assert response.decision.status == "answered"
    assert decision.decided_by_user_id == test_user.id and decision.reason == "Proceed"
    assert response.process is None


async def test_designated_agent_authority_cannot_be_answered_by_human(
    db_session, conversation_goal_run, test_user, test_agent
):
    goal, _ = conversation_goal_run
    decision = await OrchestrationAuthorityDecisionService().create_pending(
        db_session, goal.id, decision_key="agent-chat-test", title="Choose", question="Continue?",
        options=["approve"], authority="agent", authority_agent_id=test_agent.id,
    )
    with pytest.raises(HTTPException) as error:
        await _answer(db_session, goal, decision, test_user, selected_option="approve")
    assert error.value.status_code == 409 and decision.status == "pending"


async def test_runtime_version_and_duplicate_protections(db_session, test_project, test_user):
    _, goal, run, decision = await _runtime_decision(db_session, test_project)
    with pytest.raises(HTTPException) as error:
        await _answer(db_session, goal, decision, test_user, selected_option="approve")
    assert error.value.status_code == 400
    stale = await _answer(db_session, goal, decision, test_user, selected_option="approve", contract_version="old")
    assert stale.continuation_applied is False and decision.status == "pending"
    first = await _answer(db_session, goal, decision, test_user, selected_option="approve", contract_version="plan:1")
    duplicate = await _answer(db_session, goal, decision, test_user, selected_option="approve", contract_version="plan:1")
    assert first.continuation_applied is True and duplicate.continuation_applied is False
    with pytest.raises(HTTPException) as error:
        await _answer(db_session, goal, decision, test_user, selected_option="reject", contract_version="plan:1")
    assert error.value.status_code == 409
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id,
        OrchestrationAction.idempotency_key == f"run:{run.id}:kind:decision_continuation:decision:{decision.id}",
    )) == 1


async def test_runtime_answer_records_without_continuing_paused_run(db_session, test_project, test_user):
    _, goal, run, decision = await _runtime_decision(db_session, test_project)
    goal.status = run.status = "paused"
    result = await _answer(db_session, goal, decision, test_user, selected_option="approve", contract_version="plan:1")
    assert result.decision.status == "answered" and result.continuation_applied is False
    assert decision.continuation_action_id is None


async def test_recovery_approve_reserves_once_without_effects(
    db_session, goal_run, test_agent, test_user, fail_on_effect
):
    goal, run = goal_run
    _, _, _, decision = await diagnosed(db_session, goal, run, test_agent)
    first = await _answer(db_session, goal, decision, test_user, selected_option="approve_fix")
    again = await _answer(db_session, goal, decision, test_user, selected_option="approve_fix")
    assert first.decision.recovery.status == again.decision.recovery.status == "applying"
    assert first.process is None and len(await actions_of(db_session, run, "recovery_apply")) == 1
    with pytest.raises(HTTPException) as error:
        await _answer(db_session, goal, decision, test_user, selected_option="reject", reason="changed")
    assert error.value.status_code == 409


async def test_recovery_rejection_requires_reason_and_preserves_proposal(
    db_session, goal_run, test_agent, test_user, fail_on_effect
):
    goal, run = goal_run
    _, _, proposal, decision = await diagnosed(db_session, goal, run, test_agent)
    with pytest.raises(HTTPException) as error:
        await _answer(db_session, goal, decision, test_user, selected_option="reject", reason=" ")
    assert error.value.status_code == 400 and decision.status == "pending"
    result = await _answer(db_session, goal, decision, test_user, selected_option="reject", reason="Too risky")
    assert result.decision.recovery.status == "rejected"
    assert proposal.status == "failed" and proposal.error == "rejected_by_human"
    assert await actions_of(db_session, run, "recovery_apply") == []


async def test_recovery_drift_cannot_reserve_effect(db_session, goal_run, test_agent, test_user, fail_on_effect, sessions):
    goal, run = goal_run
    _, _, proposal, decision = await diagnosed(db_session, goal, run, test_agent)
    proposal.request = {**proposal.request, "problem": "Different reviewed problem"}
    with pytest.raises(HTTPException) as error:
        await _answer(db_session, goal, decision, test_user, selected_option="approve_fix")
    assert error.value.status_code == 409
    async with sessions() as fresh:
        saved = await fresh.get(OrchestrationAction, proposal.id)
        assert saved.status == "failed" and saved.error == "state_changed"
        assert await actions_of(fresh, run, "recovery_apply") == []


@pytest.mark.parametrize("case,code", [("incomplete", 409), ("duplicate", 400), ("blank_edit", 400)])
async def test_batch_cannot_bypass_current_complete_set_and_edit_rules(
    db_session, review_goal, review_run, test_user, case, code
):
    review_goal.status, review_run.status = "active", "running"
    agents, _, _, current, decisions = await _three_pending_proposals(db_session, review_goal, review_run)
    answers = [{"decision_id": decision.id, "selected_option": "approve"} for decision in decisions]
    if case == "incomplete":
        answers.pop()
    elif case == "duplicate":
        answers[1] = answers[0]
    else:
        answers[0] = {**answers[0], "selected_option": "edit", "edited_description": " ", "edited_persona": "Useful"}
    with pytest.raises(HTTPException) as error:
        await application.answer_agent_review_batch(db_session, review_goal.project_id, review_goal.id,
            OrchestrationAgentDefinitionReviewBatchAnswerRequest(answers=answers), test_user)
    assert error.value.status_code == code
    assert all(decision.status == "pending" for decision in decisions)
    assert all(agent.description == "Original description." for agent in agents)
    assert current.status == "waiting_decision"


async def _definition_answer(sessions, seeded):
    async with sessions() as db:
        result = await application.answer_agent_review_batch(db, seeded.project_id, seeded.goal_id,
            OrchestrationAgentDefinitionReviewBatchAnswerRequest(answers=[{
                "decision_id": seeded.decision_id, "selected_option": "approve"}]),
            await db.get(User, seeded.user_id))
        await db.commit()
        return result


async def _hierarchy_answer(sessions, seeded, option):
    data = {"selected_option": option}
    if option == "edit":
        data["edited_agent"] = {**proposed_agent_definition()["definition"], "name": f"edited-{seeded.proposal_name}"}
    async with sessions() as db:
        result = await application.answer_decision(db, seeded.project_id, seeded.goal_id, seeded.decision_id,
            OrchestrationDecisionAnswerRequest(**data), await db.get(User, seeded.user_id))
        await db.commit()
        return result


async def test_batch_answers_reserve_then_apply_roster_first(sessions, tmp_path, approve_semantic_reviews, lock_tracer):
    seeded = await seed_definition_proposal(sessions, tmp_path)
    lock_tracer.start()
    result = await _definition_answer(sessions, seeded)
    assert_entry_handoff(lock_tracer, seeded.goal_id)
    single_agent_write(lock_tracer, agent_id=seeded.agent_id)
    assert (await fresh_agent(sessions, seeded.agent_id)).description == PROPOSED_DESCRIPTION
    assert result.process["status"] == "completed"


async def test_batch_handoff_does_not_resume_replacement_run(sessions, tmp_path, approve_semantic_reviews, lock_tracer):
    seeded = await seed_definition_proposal(sessions, tmp_path)
    lock_tracer.start()
    interfere_after_roster_release(lock_tracer, sessions, seeded, "run_replaced")
    await outcome_or_conflict(_definition_answer(sessions, seeded))
    single_agent_write(lock_tracer, agent_id=seeded.agent_id)
    assert (await current_process(sessions, seeded.goal_id, "agent_definition_review")).status != "completed"


@pytest.mark.parametrize("option", ["approve", "edit"])
async def test_hierarchy_proposal_reserves_and_creates_roster_first(sessions, tmp_path, monkeypatch, lock_tracer, option):
    seeded = await seed_hierarchy_proposal(sessions, tmp_path, monkeypatch)
    lock_tracer.start()
    await _hierarchy_answer(sessions, seeded, option)
    assert_entry_handoff(lock_tracer, seeded.goal_id)
    name = seeded.proposal_name if option == "approve" else f"edited-{seeded.proposal_name}"
    single_agent_write(lock_tracer, name=name)
    created = await agents_named(sessions, name)
    assert len(created) == 1
    answered = next(item for item in await decisions_for(sessions, seeded.goal_id) if item.id == seeded.decision_id)
    assert answered.status == "answered"
    if option == "edit":
        assert json.loads(answered.reason)["edited_agent"]["name"] == name


async def test_hierarchy_drift_leaves_proposal_pending(sessions, tmp_path, monkeypatch):
    seeded = await seed_hierarchy_proposal(sessions, tmp_path, monkeypatch)
    async with sessions() as db:
        goal = await db.get(OrchestrationGoal, seeded.goal_id)
        goal.objective = "Changed work scope"
        await db.commit()
    with pytest.raises(HTTPException) as error:
        await _hierarchy_answer(sessions, seeded, "approve")
    assert error.value.status_code == 409
    decisions = await decisions_for(sessions, seeded.goal_id)
    assert next(item for item in decisions if item.id == seeded.decision_id).status == "pending"
    assert await agents_named(sessions, seeded.proposal_name) == []


async def test_hierarchy_handoff_revalidates_revoked_approval(sessions, tmp_path, monkeypatch, lock_tracer):
    seeded = await seed_hierarchy_proposal(sessions, tmp_path, monkeypatch)
    lock_tracer.start()
    interfere_after_roster_release(lock_tracer, sessions, seeded, "approval_revoked")
    await outcome_or_conflict(_hierarchy_answer(sessions, seeded, "approve"))
    single_agent_write(lock_tracer, name=seeded.proposal_name)
    assert not any(item.decision_key == "team_hierarchy:approval"
                   for item in await decisions_for(sessions, seeded.goal_id))
