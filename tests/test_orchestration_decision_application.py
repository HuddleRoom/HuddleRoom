"""Shared application retains the complete specialized decision paths."""
import json
import uuid
import pytest
from fastapi import HTTPException
from sqlalchemy import func, select

from huddleroom.models import User
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGoal, OrchestrationRun
from huddleroom.models.project import Project
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
from huddleroom.schemas.orchestration import (
    OrchestrationAgentDefinitionReviewBatchAnswerRequest, OrchestrationDecisionAnswerRequest,
)
from huddleroom.services import orchestration_decision_application as application
from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
from huddleroom.services.orchestration_plan_review import PlanReviewService
from tests.test_orchestration_supervision_authority import _runtime_decision
from tests.test_orchestration_supervision_authority import _text_runtime_decision
from huddleroom.models.event_log import EventLog
from tests.test_orchestration_recovery_authority import (
    actions_of, diagnosed, fail_on_effect, goal_run, db_session, _owned_diagnosis_setup,
)
from tests.test_orchestration_agent_definition_review import _three_pending_proposals, review_goal, review_run
from tests.test_orchestration_roster_mutation import (
    PROPOSED_DESCRIPTION, agents_named, approve_semantic_reviews, assert_entry_handoff, current_process,
    decisions_for, fresh_agent, interfere_after_roster_release, lock_tracer, outcome_or_conflict,
    proposed_agent_definition, seed_definition_proposal, seed_hierarchy_proposal, sessions, single_agent_write,
)
from tests.test_orchestration_roadmap_plans import _roadmap_acceptance

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("safe_goal_analysis")]


async def _answer(db, goal, decision, user, **data):
    return await application.answer_decision(db, goal.project_id, goal.id, decision.id,
                                            OrchestrationDecisionAnswerRequest(**data), user)


async def test_g3_stale_roadmap_answer_stays_stale_when_bytes_are_restored_before_rollback_marker(
    sessions, tmp_path, monkeypatch,
):
    workspace = tmp_path / "g3-stale-review"
    workspace.mkdir()
    async with sessions() as setup:
        project = Project(name="G3 stale review", description="", workspace_path=str(workspace), config={})
        user = User(email="g3-stale@example.com", hashed_password="x", display_name="G3", role="member")
        setup.add_all([project, user])
        await setup.flush()
        _, goal, run, artifact, decision, _ = await _roadmap_acceptance(setup, project, user)
        review = await PlanReviewService().ensure(setup, goal, run, artifact)
        export = workspace / review.export_path
        original = export.read_text(encoding="utf-8")
        identifiers = project.id, goal.id, run.id, artifact.id, decision.id, user.id
        await setup.commit()

    export.write_text("changed after review", encoding="utf-8")
    original_preserve = application._preserve_stale_roadmap_plan_answer

    async def restore_then_preserve(db, decision_id, run_id, goal_id, stale_record):
        export.write_text(original, encoding="utf-8")
        async with sessions() as concurrent:
            concurrent_run = await concurrent.get(OrchestrationRun, run_id)
            concurrent_run.plan_state = {**concurrent_run.plan_state, "concurrent_control": "kept"}
            concurrent_run.control_version += 1
            await concurrent.commit()
        await original_preserve(db, decision_id, run_id, goal_id, stale_record)

    monkeypatch.setattr(application, "_preserve_stale_roadmap_plan_answer", restore_then_preserve)
    async with sessions() as submit:
        project_id, goal_id, _, _, decision_id, _ = identifiers
        with pytest.raises(HTTPException, match="Roadmap plan") as error:
            await application.answer_decision(
                submit, project_id, goal_id, decision_id,
                OrchestrationDecisionAnswerRequest(selected_option="approve", reason="Approve candidate."), user,
            )
        assert error.value.status_code == 409

    async with sessions() as fresh:
        _, _, run_id, artifact_id, decision_id, _ = identifiers
        saved_run = await fresh.get(OrchestrationRun, run_id)
        saved = await fresh.get(OrchestrationAuthorityDecision, decision_id)
        record = saved_run.plan_state["plan_reviews"][str(artifact_id)]
        assert record["state"] == "stale"
        assert saved_run.plan_state["concurrent_control"] == "kept"
        assert (saved.status, saved.selected_option) == ("pending", None)
    async with sessions() as retry:
        with pytest.raises(HTTPException, match="Roadmap plan"):
            await application.answer_decision(
                retry, identifiers[0], identifiers[1], decision_id,
                OrchestrationDecisionAnswerRequest(selected_option="approve", reason="Approve candidate."), user,
            )


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
    assert response.answer_receipt.action_id is not None
    assert response.answer_receipt.run_id == run.id


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


async def test_manager_authority_decision_rejects_human_http_answer(
    db_session, client, auth_headers, test_project, conversation_goal_run, test_agent
):
    goal, _ = conversation_goal_run
    decision = await OrchestrationAuthorityDecisionService().create_pending(
        db_session, goal.id, decision_key="manager-http-test", title="Choose", question="Continue?",
        options=["approve"], authority="manager", authority_agent_id=test_agent.id,
    )
    await db_session.commit()
    response = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/decisions/{decision.id}/answer",
        headers=auth_headers, json={"selected_option": "approve", "reason": "human answer"},
    )
    assert 400 <= response.status_code < 500, response.text
    await db_session.refresh(decision)
    assert decision.status == "pending"


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
    async with sessions() as consumer:
        await application.drain_answer_applications(consumer, seeded.run_id)
        await consumer.commit()
    return result


async def _hierarchy_answer(sessions, seeded, option):
    data = {"selected_option": option}
    if option == "edit":
        data["edited_agent"] = {**proposed_agent_definition()["definition"], "name": f"edited-{seeded.proposal_name}"}
    async with sessions() as db:
        result = await application.answer_decision(db, seeded.project_id, seeded.goal_id, seeded.decision_id,
            OrchestrationDecisionAnswerRequest(**data), await db.get(User, seeded.user_id))
        await db.commit()
    async with sessions() as consumer:
        await application.drain_answer_applications(consumer, seeded.run_id)
        await consumer.commit()
    return result


async def test_batch_answers_reserve_then_apply_roster_first(sessions, tmp_path, approve_semantic_reviews, lock_tracer):
    seeded = await seed_definition_proposal(sessions, tmp_path)
    lock_tracer.start()
    result = await _definition_answer(sessions, seeded)
    assert_entry_handoff(lock_tracer, seeded.goal_id)
    single_agent_write(lock_tracer, agent_id=seeded.agent_id)
    assert (await fresh_agent(sessions, seeded.agent_id)).description == PROPOSED_DESCRIPTION
    assert result.process is None and result.answer_receipt.application_state == "pending"
    assert (await current_process(sessions, seeded.goal_id, "agent_definition_review")).status == "completed"


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


async def _spr231_committed_ordinary(sessions, tmp_path):
    from tests.test_orchestration_roster_mutation import seed_project_user
    from huddleroom.models.orchestration import OrchestrationRun
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService
    async with sessions() as db:
        project, user = await seed_project_user(db, tmp_path)
        goal = OrchestrationGoal(project_id=project.id, objective="Saved answer survives request loss")
        db.add(goal)
        await db.flush()
        run = OrchestrationRun(goal_id=goal.id, phase="baseline", status="running", baseline_authorized=True)
        db.add(run)
        await db.flush()
        processes = OrchestrationProcessService()
        source = await processes.start_process(db, goal.id, process_type="goal_definition", trigger_reason="SPR231", run_id=run.id)
        await processes.park_process(db, source)
        decision = await OrchestrationAuthorityDecisionService().create_pending(
            db, goal.id, decision_key="goal_definition:adaptive:1:0:objective", title="Approve goal", question="Approve the reviewed goal?",
            authority="human", options=["approve", "request_changes"], run_id=run.id, source_process_run_id=source.id)
        await db.commit()
        return project, user, goal, run, source, decision


@pytest.mark.parametrize("selected_option", ["approve", "request_changes"])
async def test_spr231_ordinary_answer_returns_saved_before_continuation(sessions, tmp_path, monkeypatch, selected_option):
    from huddleroom.services.orchestration_debug_service import OrchestrationDebugService
    project, user, goal, run, source, decision = await _spr231_committed_ordinary(sessions, tmp_path)
    async def forbidden(*args, **kwargs):
        pytest.fail("ordinary answer acceptance awaited source/model continuation")
    monkeypatch.setattr(OrchestrationDebugService, "advance_process", forbidden)
    from huddleroom.services.orchestration_service import OrchestrationService
    def forbidden_mutex(*args, **kwargs):
        pytest.fail("ordinary acceptance acquired a long transition mutex")
    monkeypatch.setattr(OrchestrationService, "_lock_goal_for_baseline_transition", forbidden_mutex)
    async with sessions() as submitting:
        response = await application.answer_decision(
            submitting, project.id, goal.id, decision.id,
            OrchestrationDecisionAnswerRequest(selected_option=selected_option, reason="Reviewed exact scope"), user)
        assert response.process is None
        assert response.answer_receipt.application_state == "pending"
        application_id = response.answer_receipt.action_id
    async with sessions() as fresh:
        saved = await fresh.get(OrchestrationAuthorityDecision, decision.id)
        obligation = await fresh.get(OrchestrationAction, application_id)
        assert saved.status == "answered" and saved.selected_option == selected_option
        assert saved.reason == "Reviewed exact scope"
        assert saved.continuation_action_id == obligation.id
        assert obligation.action_type == "apply_authority_answer" and obligation.status == "reserved"
        assert await fresh.scalar(select(func.count()).select_from(EventLog).where(
            EventLog.dedup_key == f"authority.decision_resolved:{decision.id}")) == 1


async def test_spr231_locked_database_during_drain_stays_pending_then_applies_once(sessions, tmp_path, monkeypatch):
    from sqlalchemy.exc import OperationalError
    from huddleroom.services.orchestration_debug_service import OrchestrationDebugService
    project, user, goal, run, source, decision = await _spr231_committed_ordinary(sessions, tmp_path)
    original, attempts = OrchestrationDebugService.advance_process, []
    async def locked_once(service, db, goal, run, process_type, *args, **kwargs):
        attempts.append(process_type)
        if len(attempts) == 1:
            raise OperationalError("database is locked", None, None)
        return await original(service, db, goal, run, process_type, *args, **kwargs)
    monkeypatch.setattr(OrchestrationDebugService, "advance_process", locked_once)
    async with sessions() as submitting:
        result = await application.answer_decision(submitting, project.id, goal.id, decision.id,
            OrchestrationDecisionAnswerRequest(selected_option="approve", reason="Retry after lock"), user)
        action_id = result.answer_receipt.action_id
    async with sessions() as consumer:
        await application.drain_answer_applications(consumer, run.id)
        await consumer.commit()
    async with sessions() as fresh:
        obligation = await fresh.get(OrchestrationAction, action_id, populate_existing=True)
        assert obligation.status == "reserved"
        assert obligation.dispatch_contract["application_state"] == "pending"
    async with sessions() as consumer:
        await application.drain_answer_applications(consumer, run.id)
        await consumer.commit()
    assert len(attempts) == 2
    async with sessions() as fresh:
        obligation = await fresh.get(OrchestrationAction, action_id, populate_existing=True)
        assert obligation.status == "completed"
        assert obligation.dispatch_contract["application_state"] == "applied"
        assert await fresh.scalar(select(func.count()).select_from(EventLog).where(
            EventLog.dedup_key == f"authority.decision_resolved:{decision.id}")) == 1


async def test_spr248_stalled_source_blocks_saved_answer_without_continuation_timestamp(
    sessions, tmp_path, monkeypatch,
):
    from huddleroom.services.orchestration_debug_service import OrchestrationDebugService
    from huddleroom.models.orchestration_process import OrchestrationProcessRun

    project, user, goal, run, source, decision = await _spr231_committed_ordinary(sessions, tmp_path)

    async def stalled(*_args, **_kwargs):
        return {"status": "waiting_decision"}

    monkeypatch.setattr(OrchestrationDebugService, "advance_process", stalled)
    async with sessions() as submitting:
        accepted = await application.answer_decision(
            submitting, project.id, goal.id, decision.id,
            OrchestrationDecisionAnswerRequest(selected_option="approve", reason="Await actual source progress"), user,
        )
    async with sessions() as consumer:
        result = await application.drain_answer_applications(consumer, run.id)
        await consumer.commit()
    assert accepted.answer_receipt.action_id in result.failed
    async with sessions() as fresh:
        action = await fresh.get(OrchestrationAction, accepted.answer_receipt.action_id)
        saved = await fresh.get(OrchestrationAuthorityDecision, decision.id)
        assert action.status == "failed"
        assert action.dispatch_contract["application_state"] == "blocked"
        assert action.dispatch_contract["reason"] == "source_process_did_not_advance"
        assert saved.continuation_applied_at is None
        assert (await fresh.get(OrchestrationProcessRun, source.id)).status == "waiting_decision"


async def test_spr248_linked_successor_applies_request_changes_answer(sessions, tmp_path, monkeypatch):
    from huddleroom.services.orchestration_debug_service import OrchestrationDebugService
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    project, user, goal, run, source, decision = await _spr231_committed_ordinary(sessions, tmp_path)

    async def supersede(_service, db, current_goal, current_run, process_type):
        return await OrchestrationProcessService().start_process(
            db, current_goal.id, process_type=process_type, trigger_reason="requested revision",
            run_id=current_run.id, supersede_waiting=True,
        )

    monkeypatch.setattr(OrchestrationDebugService, "advance_process", supersede)
    async with sessions() as submitting:
        accepted = await application.answer_decision(
            submitting, project.id, goal.id, decision.id,
            OrchestrationDecisionAnswerRequest(selected_option="request_changes", reason="Revise the objective"), user,
        )
    async with sessions() as consumer:
        result = await application.drain_answer_applications(consumer, run.id)
        await consumer.commit()
    assert accepted.answer_receipt.action_id in result.applied
    async with sessions() as fresh:
        action = await fresh.get(OrchestrationAction, accepted.answer_receipt.action_id)
        saved = await fresh.get(OrchestrationAuthorityDecision, decision.id)
        current = await OrchestrationProcessService().get_current(fresh, goal.id, source.process_type)
        assert action.status == "completed" and action.dispatch_contract["application_state"] == "applied"
        assert saved.continuation_applied_at is not None
        assert current is not None and current.id != source.id and current.run_id == run.id


@pytest.mark.parametrize("forgery", ["missing", "wrong_type", "wrong_key", "foreign_goal", "stale_source"])
async def test_spr248_forged_hierarchy_continuation_effect_cannot_settle_answer(
    sessions, tmp_path, monkeypatch, forgery,
):
    seeded = await seed_hierarchy_proposal(sessions, tmp_path, monkeypatch)
    async with sessions() as submitting:
        goal = await submitting.get(OrchestrationGoal, seeded.goal_id)
        user = await submitting.get(User, seeded.user_id)
        accepted = await application.answer_decision(
            submitting, seeded.project_id, seeded.goal_id, seeded.decision_id,
            OrchestrationDecisionAnswerRequest(selected_option="approve"), user,
        )
        action = await submitting.get(OrchestrationAction, accepted.answer_receipt.action_id)
        decision = await submitting.get(OrchestrationAuthorityDecision, seeded.decision_id)
        target_id = uuid.uuid4()
        target_type = "decision"
        if forgery == "wrong_type":
            target_type = "process"
        elif forgery == "wrong_key":
            target_id = decision.id
        elif forgery == "foreign_goal":
            from huddleroom.models.orchestration import OrchestrationRun
            foreign_goal = OrchestrationGoal(project_id=goal.project_id, objective="Foreign hierarchy target")
            submitting.add(foreign_goal)
            await submitting.flush()
            foreign_run = OrchestrationRun(goal_id=foreign_goal.id, status="running", phase="baseline")
            submitting.add(foreign_run)
            await submitting.flush()
            foreign = OrchestrationAuthorityDecision(
                goal_id=foreign_goal.id, run_id=foreign_run.id, decision_key="team_hierarchy:approval",
                title="Foreign hierarchy approval", authority="human", question="Approve?", options=[{"key": "approve"}],
            )
            submitting.add(foreign)
            await submitting.flush()
            target_id = foreign.id
        elif forgery == "stale_source":
            from huddleroom.services.orchestration_process_service import OrchestrationProcessService
            await OrchestrationProcessService().start_process(
                submitting, goal.id, process_type="team_hierarchy", trigger_reason="stale replacement",
                run_id=seeded.run_id, supersede_waiting=True,
            )
        effects = dict(action.dispatch_contract["effects"])
        effects[f"continuation_decision:{decision.source_process_run_id}"] = {
            "target_type": target_type,
            "target_id": str(target_id),
        }
        action.dispatch_contract = {**action.dispatch_contract, "effects": effects}
        await submitting.commit()
    async with sessions() as consumer:
        result = await application.drain_answer_applications(consumer, seeded.run_id)
        await consumer.commit()
    assert accepted.answer_receipt.action_id not in result.applied
    async with sessions() as fresh:
        action = await fresh.get(OrchestrationAction, accepted.answer_receipt.action_id)
        decision = await fresh.get(OrchestrationAuthorityDecision, seeded.decision_id)
        assert action.status == "failed"
        assert decision.continuation_applied_at is None


async def test_spr231_non_lock_operational_error_during_drain_fails_without_retry(sessions, tmp_path, monkeypatch):
    from sqlalchemy.exc import OperationalError
    from huddleroom.services.orchestration_debug_service import OrchestrationDebugService
    project, user, goal, run, source, decision = await _spr231_committed_ordinary(sessions, tmp_path)
    attempts = []
    async def missing_table(service, db, goal, run, process_type, *args, **kwargs):
        attempts.append(process_type)
        raise OperationalError("no such table: x", None, None)
    monkeypatch.setattr(OrchestrationDebugService, "advance_process", missing_table)
    async with sessions() as submitting:
        result = await application.answer_decision(submitting, project.id, goal.id, decision.id,
            OrchestrationDecisionAnswerRequest(selected_option="approve", reason="Non-lock failure"), user)
        action_id = result.answer_receipt.action_id
    async with sessions() as consumer:
        await application.drain_answer_applications(consumer, run.id)
        await consumer.commit()
    async with sessions() as fresh:
        obligation = await fresh.get(OrchestrationAction, action_id, populate_existing=True)
        assert obligation.status == "failed"
        assert obligation.dispatch_contract["application_state"] == "failed"
    async with sessions() as consumer:
        await application.drain_answer_applications(consumer, run.id)
        await consumer.commit()
    assert len(attempts) == 1


@pytest.mark.parametrize("selected_option", ["approve", "request_changes"])
async def test_spr231_request_loss_fresh_consumer_applies_original_answer_once(sessions, tmp_path, monkeypatch, selected_option):
    from huddleroom.services.orchestration_debug_service import OrchestrationDebugService
    from huddleroom.models.orchestration_process import OrchestrationProcessRun
    project, user, goal, run, source, decision = await _spr231_committed_ordinary(sessions, tmp_path)
    original, advances = OrchestrationDebugService.advance_process, []
    async def advance(service, db, goal, run, process_type, *args, **kwargs):
        advances.append((goal.id, run.id, process_type))
        return await original(service, db, goal, run, process_type, *args, **kwargs)
    monkeypatch.setattr(OrchestrationDebugService, "advance_process", advance)
    async with sessions() as submitting:
        # No caller commit, no queued request coroutine survives session close.
        result = await application.answer_decision(submitting, project.id, goal.id, decision.id,
            OrchestrationDecisionAnswerRequest(selected_option=selected_option, reason="Keep exact reviewed payload"), user)
        action_id = result.answer_receipt.action_id
        assert result.answer_receipt.application_state == "pending"
    assert advances == []
    async with sessions() as consumer:
        await application.drain_answer_applications(consumer, run.id)
        await consumer.commit()
    async with sessions() as fresh:
        marker = await fresh.get(OrchestrationAction, action_id)
        assert marker.status == "completed"
        process = await fresh.get(OrchestrationProcessRun, source.id)
        snapshot = (process.status, process.outputs, process.completed_at)
    assert advances == [(goal.id, run.id, "goal_definition")]
    async with sessions() as consumer:
        await application.drain_answer_applications(consumer, run.id)
        await consumer.commit()
    # Exact replay succeeds after source advance without calling the source again.
    async with sessions() as replay:
        result = await application.answer_decision(replay, project.id, goal.id, decision.id,
            OrchestrationDecisionAnswerRequest(selected_option=selected_option, reason="Keep exact reviewed payload"), user)
        assert result.answer_receipt.action_id == action_id
        assert result.answer_receipt.application_state == "applied"
    assert advances == [(goal.id, run.id, "goal_definition")]
    async with sessions() as fresh:
        process = await fresh.get(OrchestrationProcessRun, source.id)
        assert (process.status, process.outputs, process.completed_at) == snapshot
        assert await fresh.scalar(select(func.count()).select_from(EventLog).where(
            EventLog.dedup_key == f"authority.decision_resolved:{decision.id}")) == 1


@pytest.mark.parametrize("selected_option", ["approve", "request_changes"])
async def test_spr231_concurrent_exact_replays_one_obligation_and_conflicting_payload_stays_409(sessions, tmp_path, selected_option):
    import asyncio
    project, user, goal, run, source, decision = await _spr231_committed_ordinary(sessions, tmp_path)
    async def submit(data=None, actor=None):
        async with sessions() as db:
            return await application.answer_decision(db, project.id, goal.id, decision.id,
                OrchestrationDecisionAnswerRequest(**(data or {"selected_option": selected_option, "reason": "Exact reviewed reason"})),
                actor or user)
    results = await asyncio.gather(submit(), submit(), submit())
    assert len({r.answer_receipt.action_id for r in results}) == 1
    assert all(r.answer_receipt.application_state == "pending" for r in results)
    async with sessions() as db:
        other = User(email=f"other-{uuid.uuid4()}@test.invalid", hashed_password="test")
        db.add(other)
        await db.commit()
    for changed, actor in [
        ({"selected_option": "request_changes" if selected_option == "approve" else "approve", "reason": "Exact reviewed reason"}, user),
        ({"selected_option": selected_option, "reason": "Different reason"}, user),
        ({"selected_option": selected_option, "reason": "Exact reviewed reason", "contract_version": "changed-contract"}, user),
        (None, other),
    ]:
        with pytest.raises(HTTPException) as error:
            await submit(changed, actor)
        assert error.value.status_code == 409
    async with sessions() as db:
        saved = await db.get(OrchestrationAuthorityDecision, decision.id)
        assert (saved.selected_option, saved.reason, saved.decided_by_user_id) == (selected_option, "Exact reviewed reason", user.id)
        obligations = list((await db.scalars(select(OrchestrationAction).where(
            OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == "apply_authority_answer"))).all())
        assert len(obligations) == 1
        assert await db.scalar(select(func.count()).select_from(EventLog).where(
            EventLog.dedup_key == f"authority.decision_resolved:{decision.id}")) == 1


@pytest.mark.parametrize("explicit_caller", [False, True])
@pytest.mark.parametrize("failed_callback", ["validation", "reservation", "completion"])
async def test_spr231_callback_failure_rolls_back_acceptance_but_preserves_caller_work(sessions, tmp_path, explicit_caller, failed_callback):
    from sqlalchemy import update
    from huddleroom.models.project import Project
    project, user, goal, run, source, decision = await _spr231_committed_ordinary(sessions, tmp_path)
    async with sessions() as db:
        original_root = None
        if explicit_caller:
            await db.begin()
            await db.execute(update(Project).where(Project.id == project.id).values(description="unrelated caller work"))
            original_root = db.sync_session.get_transaction()
        async def validation(db):
            assert (await db.get(OrchestrationAuthorityDecision, decision.id)).status == "pending"
            if failed_callback == "validation":
                raise RuntimeError("SPR231 callback failure")
        async def reservation(db, linkage):
            assert (await db.get(OrchestrationAuthorityDecision, decision.id)).status == "answered"
            if failed_callback == "reservation":
                raise RuntimeError("SPR231 callback failure")
        async def completion(db, response):
            assert response.answer_receipt.application_state == "pending"
            if failed_callback == "completion":
                raise RuntimeError("SPR231 callback failure")
        with pytest.raises(RuntimeError, match="SPR231 callback failure"):
            await application.answer_decision(db, project.id, goal.id, decision.id,
                OrchestrationDecisionAnswerRequest(selected_option="approve", reason="Exact reason"), user,
                validation_callback=validation, reservation_callback=reservation, completion_callback=completion)
        if explicit_caller:
            assert db.sync_session.get_transaction() is original_root and original_root.is_active
            assert await db.scalar(select(Project.description).where(Project.id == project.id)) == "unrelated caller work"
            await db.commit()
    async with sessions() as fresh:
        saved = await fresh.get(OrchestrationAuthorityDecision, decision.id)
        assert (saved.status, saved.selected_option, saved.reason, saved.continuation_action_id) == ("pending", None, None, None)
        assert await fresh.scalar(select(func.count()).select_from(EventLog).where(
            EventLog.dedup_key == f"authority.decision_resolved:{decision.id}")) == 0
        assert await fresh.scalar(select(func.count()).select_from(OrchestrationAction).where(
            OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == "apply_authority_answer")) == 0
        assert await fresh.scalar(select(Project.description).where(Project.id == project.id)) == (
            "unrelated caller work" if explicit_caller else project.description)


async def test_spr231_saved_receipt_readback_survives_pending_queue_removal_and_scopes_target(sessions, tmp_path):
    from httpx import ASGITransport, AsyncClient
    from huddleroom.main import create_app
    from huddleroom.database import get_db
    from huddleroom.dependencies import get_current_user
    project, user, goal, run, source, decision = await _spr231_committed_ordinary(sessions, tmp_path)
    app = create_app()
    async def request_db():
        async with sessions() as db:
            yield db
            await db.commit()
    async def actor():
        return user
    app.dependency_overrides[get_db], app.dependency_overrides[get_current_user] = request_db, actor
    base = f"/api/v1/projects/{project.id}/orchestration/goals/{goal.id}/decisions"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        accepted = await client.post(f"{base}/{decision.id}/answer", json={"selected_option": "request_changes", "reason": "Reviewed revision"})
        assert accepted.status_code == 200, accepted.text
        receipt = accepted.json()["answer_receipt"]
        assert receipt["decision_ids"] == [str(decision.id)]
        assert receipt["actor_user_id"] == str(user.id)
        assert receipt["source_process_run_id"] == str(source.id) and receipt["run_id"] == str(run.id)
        queue = await client.get(base, params={"status": "pending"})
        assert queue.status_code == 200 and queue.json() == []
        readback = await client.get(f"{base}/{decision.id}")
        assert readback.status_code == 200, readback.text
        assert readback.json()["answer_receipt"] == receipt
        assert readback.json()["decision"]["reason"] == "Reviewed revision"
        foreign_project = await client.get(f"/api/v1/projects/{uuid.uuid4()}/orchestration/goals/{goal.id}/decisions/{decision.id}")
        foreign_goal = await client.get(f"/api/v1/projects/{project.id}/orchestration/goals/{uuid.uuid4()}/decisions/{decision.id}")
        missing = await client.get(f"{base}/{uuid.uuid4()}")
        assert [foreign_project.status_code, foreign_goal.status_code, missing.status_code] == [404, 404, 404]


@pytest.mark.parametrize("startup", [False, True])
async def test_spr231_saved_answer_recovery_waits_for_resume_and_baseline_continue(sessions, tmp_path, monkeypatch, startup):
    from huddleroom.services.orchestration_service import OrchestrationService
    from huddleroom.services.orchestration_debug_service import OrchestrationDebugService
    from huddleroom.models.orchestration import OrchestrationRun
    project, user, goal, run, source, decision = await _spr231_committed_ordinary(sessions, tmp_path)
    async with sessions() as submitting:
        result = await application.answer_decision(submitting, project.id, goal.id, decision.id,
            OrchestrationDecisionAnswerRequest(selected_option="request_changes", reason="Reviewed scope"), user)
        action_id = result.answer_receipt.action_id
    service = OrchestrationService()
    async with sessions() as pausing:
        if startup:
            await service.pause_existing_goals_for_startup(pausing)
            await pausing.commit()
        else:
            await service.pause_goal(pausing, project.id, goal.id)
    original, calls = OrchestrationDebugService.advance_process, []
    async def advance(self, *args, **kwargs):
        calls.append(True)
        return await original(self, *args, **kwargs)
    monkeypatch.setattr(OrchestrationDebugService, "advance_process", advance)
    async with sessions() as consumer:
        await application.drain_answer_applications(consumer, run.id)
        await consumer.commit()
    assert calls == []
    async with sessions() as resume:
        await service.resume_goal(resume, project.id, goal.id)
        await resume.commit()
    async with sessions() as consumer:
        assert (await consumer.get(OrchestrationRun, run.id)).baseline_authorized is False
        await consumer.rollback()  # Observer read must not own the consumer invocation.
        await application.drain_answer_applications(consumer, run.id)
        await consumer.commit()
    assert calls == []
    async with sessions() as continuing:
        await service.authorize_baseline(continuing, project.id, goal.id, actor=f"human:{user.id}")
        await continuing.commit()
    async with sessions() as consumer:
        await application.drain_answer_applications(consumer, run.id)
        await consumer.commit()
    assert calls == [True]
    async with sessions() as fresh:
        assert (await fresh.get(OrchestrationAction, action_id)).status == "completed"
        assert (await fresh.get(OrchestrationAuthorityDecision, decision.id)).reason == "Reviewed scope"


async def test_spr231_accepted_answer_never_applies_to_replacement_source(sessions, tmp_path, monkeypatch):
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService
    from huddleroom.services.orchestration_debug_service import OrchestrationDebugService
    project, user, goal, run, source, decision = await _spr231_committed_ordinary(sessions, tmp_path)
    data = OrchestrationDecisionAnswerRequest(selected_option="approve", reason="Original reviewed source")
    async with sessions() as submitting:
        accepted = await application.answer_decision(submitting, project.id, goal.id, decision.id, data, user)
    async with sessions() as replacement:
        successor = await OrchestrationProcessService().start_process(
            replacement, goal.id, process_type="goal_definition", trigger_reason="Separate user rerun",
            run_id=run.id, supersede_waiting=True)
        await replacement.commit()
        assert successor.id != source.id
    async def forbidden(*args, **kwargs):
        pytest.fail("saved answer advanced a replacement source")
    monkeypatch.setattr(OrchestrationDebugService, "advance_process", forbidden)
    async with sessions() as consumer:
        await application.drain_answer_applications(consumer, run.id)
        await consumer.commit()
    async with sessions() as replay:
        saved = await application.answer_decision(replay, project.id, goal.id, decision.id, data, user)
        assert saved.answer_receipt.action_id == accepted.answer_receipt.action_id
        assert saved.answer_receipt.application_state == "blocked"
        assert saved.answer_receipt.reason
        assert saved.decision.reason == "Original reviewed source"
    async with sessions() as fresh:
        from huddleroom.models.orchestration_process import OrchestrationProcessRun
        current = await fresh.get(OrchestrationProcessRun, successor.id)
        assert current.status == "running" and current.outputs == {}
        assert await fresh.scalar(select(func.count()).select_from(EventLog).where(
            EventLog.dedup_key == f"authority.decision_resolved:{decision.id}")) == 1


@pytest.mark.parametrize("worker_path", ["committed_event", "periodic_scan"])
@pytest.mark.parametrize("legacy_null_run", [False, True])
async def test_spr231_actual_worker_recovers_saved_answer_without_request_task(sessions, tmp_path, monkeypatch, worker_path, legacy_null_run):
    from unittest.mock import AsyncMock
    from huddleroom.workers import orchestration_tasks as workers
    from huddleroom.services.orchestration_supervision_scheduler import OrchestrationSupervisionScheduler
    project, user, goal, run, source, decision = await _spr231_committed_ordinary(sessions, tmp_path)
    if legacy_null_run:
        async with sessions() as setup:
            legacy = await setup.get(OrchestrationAuthorityDecision, decision.id)
            legacy.run_id = None
            await setup.commit()
    async with sessions() as submitting:
        accepted = await application.answer_decision(submitting, project.id, goal.id, decision.id,
            OrchestrationDecisionAnswerRequest(selected_option="approve", reason="Exact worker recovery"), user)
    monkeypatch.setattr(workers, "AsyncSessionLocal", sessions)
    monkeypatch.setattr(workers.settings, "orchestration_event_coalesce_seconds", 0)
    # Isolate unrelated judgment/sweep work; actual saved-answer scan/event,
    # durable consumer, native source advance and commits remain real.
    monkeypatch.setattr(OrchestrationSupervisionScheduler, "record_event", AsyncMock(return_value=[]))
    monkeypatch.setattr(OrchestrationSupervisionScheduler, "sweep", AsyncMock(return_value=[]))
    if worker_path == "committed_event":
        async with sessions() as fresh:
            event_id = await fresh.scalar(select(EventLog.id).where(
                EventLog.dedup_key == f"authority.decision_resolved:{decision.id}"))
        await workers.evaluate_supervision_event_async(event_id)
    else:
        await workers.supervise_orchestration_async()
    async with sessions() as fresh:
        assert (await fresh.get(OrchestrationAction, accepted.answer_receipt.action_id)).status == "completed"
        assert (await fresh.get(OrchestrationAuthorityDecision, decision.id)).reason == "Exact worker recovery"


@pytest.mark.parametrize("failure_kind", ["runtime", "value", "http400", "http409"])
async def test_spr231_failed_application_keeps_saved_answer_and_does_not_blindly_repeat(sessions, tmp_path, monkeypatch, failure_kind):
    from huddleroom.services.orchestration_debug_service import OrchestrationDebugService
    project, user, goal, run, source, decision = await _spr231_committed_ordinary(sessions, tmp_path)
    data = OrchestrationDecisionAnswerRequest(selected_option="approve", reason="Saved despite native failure")
    async with sessions() as submitting:
        accepted = await application.answer_decision(submitting, project.id, goal.id, decision.id, data, user)
    # A later independent source-free obligation must not starve behind failure.
    async with sessions() as setup:
        second = await OrchestrationAuthorityDecisionService().create_pending(
            setup, goal.id, decision_key="spr231:independent", title="Independent", question="Acknowledge?",
            authority="human", options=["approve"], run_id=run.id)
        await setup.commit()
    async with sessions() as submitting:
        independent = await application.answer_decision(submitting, project.id, goal.id, second.id,
            OrchestrationDecisionAnswerRequest(selected_option="approve", reason="Independent source-free answer"), user)
    calls = []
    async def failed(*args, **kwargs):
        calls.append(True)
        detail = "SPR231 native application failure"
        if failure_kind == "http400":
            raise HTTPException(400, detail=detail)
        if failure_kind == "http409":
            raise HTTPException(409, detail=detail)
        if failure_kind == "value":
            raise ValueError(detail)
        raise RuntimeError(detail)
    monkeypatch.setattr(OrchestrationDebugService, "advance_process", failed)
    async with sessions() as consumer:
        result = await application.drain_answer_applications(consumer, run.id)
        await consumer.commit()
        assert accepted.answer_receipt.action_id in result.failed
        assert independent.answer_receipt.action_id in result.applied
    async with sessions() as retry_sweep:
        await application.drain_answer_applications(retry_sweep, run.id)
        await retry_sweep.commit()
    assert calls == [True]
    async with sessions() as replay:
        saved = await application.answer_decision(replay, project.id, goal.id, decision.id, data, user)
        assert saved.decision.status == "answered" and saved.decision.reason == data.reason
        assert saved.answer_receipt.action_id == accepted.answer_receipt.action_id
        assert saved.answer_receipt.application_state in {"failed", "blocked"}
        assert "SPR231 native application failure" in saved.answer_receipt.reason


@pytest.mark.parametrize("pause_before_retry", [False, True])
async def test_spr231_explicit_application_retry_rechecks_control_and_never_repeats_answer_callbacks(sessions, tmp_path, monkeypatch, pause_before_retry):
    from huddleroom.services.orchestration_debug_service import OrchestrationDebugService
    from huddleroom.services.orchestration_service import OrchestrationService
    project, user, goal, run, source, decision = await _spr231_committed_ordinary(sessions, tmp_path)
    callbacks = []
    async def validation(db):
        callbacks.append("validation")
    async def reservation(db, linkage):
        callbacks.append("reservation")
    async def completion(db, response):
        callbacks.append("completion")
    async with sessions() as submitting:
        accepted = await application.answer_decision(submitting, project.id, goal.id, decision.id,
            OrchestrationDecisionAnswerRequest(selected_option="approve", reason="Reviewed retry identity"), user,
            validation_callback=validation, reservation_callback=reservation, completion_callback=completion)
    original, attempts = OrchestrationDebugService.advance_process, []
    async def attempt(self, *args, **kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            raise ValueError("SPR231 deterministic first failure")
        return await original(self, *args, **kwargs)
    monkeypatch.setattr(OrchestrationDebugService, "advance_process", attempt)
    async with sessions() as consumer:
        await application.drain_answer_applications(consumer, run.id)
        await consumer.commit()
    assert callbacks == ["validation", "reservation", "completion"] and attempts == [True]
    if pause_before_retry:
        async with sessions() as pause:
            await OrchestrationService().pause_goal(pause, project.id, goal.id)
        async with sessions() as retry:
            with pytest.raises(HTTPException) as error:
                await application.retry_answer_application(retry, accepted.answer_receipt.action_id)
            assert error.value.status_code == 409
    else:
        async with sessions() as retry:
            await application.retry_answer_application(retry, accepted.answer_receipt.action_id)
            await retry.commit()
        async with sessions() as consumer:
            await application.drain_answer_applications(consumer, run.id)
            await consumer.commit()
        async with sessions() as consumer:
            await application.drain_answer_applications(consumer, run.id)
            await consumer.commit()
        assert attempts == [True, True]
    assert callbacks == ["validation", "reservation", "completion"]
    async with sessions() as fresh:
        saved = await fresh.get(OrchestrationAuthorityDecision, decision.id)
        assert saved.status == "answered" and saved.reason == "Reviewed retry identity"
        action = await fresh.get(OrchestrationAction, accepted.answer_receipt.action_id)
        assert action.status == ("failed" if pause_before_retry else "completed")
        assert await fresh.scalar(select(func.count()).select_from(EventLog).where(
            EventLog.dedup_key == f"authority.decision_resolved:{decision.id}")) == 1


@pytest.mark.parametrize("saved_before_reset", [False, True])
async def test_spr231_reset_preempts_new_acceptance_but_exact_saved_replay_remains_readable(sessions, tmp_path, saved_before_reset):
    from huddleroom.services.orchestration_baseline_runtime import reset_baseline
    project, user, goal, run, source, decision = await _spr231_committed_ordinary(sessions, tmp_path)
    data = OrchestrationDecisionAnswerRequest(selected_option="approve", reason="Exact reset-safe identity")
    accepted = None
    if saved_before_reset:
        async with sessions() as submitting:
            accepted = await application.answer_decision(submitting, project.id, goal.id, decision.id, data, user)
    async with reset_baseline(goal.id):
        async with sessions() as submitting:
            if saved_before_reset:
                replay = await application.answer_decision(submitting, project.id, goal.id, decision.id, data, user)
                assert replay.answer_receipt.action_id == accepted.answer_receipt.action_id
                assert replay.decision.status == "answered"
            else:
                with pytest.raises(HTTPException) as error:
                    await application.answer_decision(submitting, project.id, goal.id, decision.id, data, user)
                assert error.value.status_code == 409
    async with sessions() as fresh:
        saved = await fresh.get(OrchestrationAuthorityDecision, decision.id)
        assert saved.status == ("answered" if saved_before_reset else "pending")
        assert await fresh.scalar(select(func.count()).select_from(EventLog).where(
            EventLog.dedup_key == f"authority.decision_resolved:{decision.id}")) == int(saved_before_reset)
        assert await fresh.scalar(select(func.count()).select_from(OrchestrationAction).where(
            OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == "apply_authority_answer")) == int(saved_before_reset)


@pytest.mark.parametrize("explicit_caller", [False, True])
@pytest.mark.parametrize("denial", ["pause", "reset", "source_changed"])
async def test_spr231_denied_retry_releases_owned_writer_and_preserves_caller_transaction(sessions, tmp_path, monkeypatch, explicit_caller, denial):
    import sqlite3
    from contextlib import AsyncExitStack
    from sqlalchemy import update
    from huddleroom.models.project import Project
    from huddleroom.services.orchestration_baseline_runtime import reset_baseline
    from huddleroom.services.orchestration_debug_service import OrchestrationDebugService
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService
    from huddleroom.services.orchestration_service import OrchestrationService
    project, user, goal, run, source, decision = await _spr231_committed_ordinary(sessions, tmp_path)
    async with sessions() as submitting:
        accepted = await application.answer_decision(submitting, project.id, goal.id, decision.id,
            OrchestrationDecisionAnswerRequest(selected_option="approve", reason="Retry ownership"), user)
    async def fail(*args, **kwargs):
        raise ValueError("Native failure before retry")
    monkeypatch.setattr(OrchestrationDebugService, "advance_process", fail)
    async with sessions() as consumer:
        await application.drain_answer_applications(consumer, run.id)
        await consumer.commit()
    if denial == "pause":
        async with sessions() as pause:
            await OrchestrationService().pause_goal(pause, project.id, goal.id)
    elif denial == "source_changed":
        async with sessions() as changed:
            await OrchestrationProcessService().start_process(changed, goal.id, process_type="goal_definition",
                trigger_reason="Separate source replacement", run_id=run.id, supersede_waiting=True)
            await changed.commit()
    async with AsyncExitStack() as stack:
        if denial == "reset":
            await stack.enter_async_context(reset_baseline(goal.id))
        async with sessions() as retry:
            root = None
            if explicit_caller:
                await retry.begin()
                await retry.execute(update(Project).where(Project.id == project.id).values(description="Unrelated retry caller write"))
                root = retry.sync_session.get_transaction()
            with pytest.raises(HTTPException) as error:
                await application.retry_answer_application(retry, accepted.answer_receipt.action_id)
            assert error.value.status_code == 409
            if explicit_caller:
                assert retry.sync_session.get_transaction() is root and root.is_active
                assert await retry.scalar(select(Project.description).where(Project.id == project.id)) == "Unrelated retry caller write"
                async with sessions() as fresh:
                    assert await fresh.scalar(select(Project.description).where(Project.id == project.id)) != "Unrelated retry caller write"
                await retry.rollback()
            # Keep the rejected service's session OPEN while another actual
            # SQLite connection proves the writer was released immediately.
            with sqlite3.connect(sessions.kw["bind"].url.database, timeout=0.1) as independent:
                independent.execute("UPDATE projects SET updated_at=updated_at WHERE id=?", (project.id.hex,))
            if not explicit_caller:
                assert not retry.in_transaction()
    async with sessions() as fresh:
        assert (await fresh.get(OrchestrationAction, accepted.answer_receipt.action_id)).status == "failed"
        assert (await fresh.get(OrchestrationAuthorityDecision, decision.id)).reason == "Retry ownership"


async def test_spr231_legacy_null_decision_run_uses_original_source_run_for_acceptance_and_replay(sessions, tmp_path):
    from huddleroom.models.orchestration import OrchestrationRun
    project, user, goal, run, source, decision = await _spr231_committed_ordinary(sessions, tmp_path)
    async with sessions() as setup:
        saved = await setup.get(OrchestrationAuthorityDecision, decision.id)
        saved.run_id = None
        newest = OrchestrationRun(goal_id=goal.id, status="completed", phase="completed")
        setup.add(newest)
        await setup.commit()
    data = OrchestrationDecisionAnswerRequest(selected_option="approve", reason="Original legacy source")
    async with sessions() as submitting:
        accepted = await application.answer_decision(submitting, project.id, goal.id, decision.id, data, user)
        assert accepted.answer_receipt.run_id == run.id
    async with sessions() as replay:
        again = await application.answer_decision(replay, project.id, goal.id, decision.id, data, user)
        assert again.answer_receipt.action_id == accepted.answer_receipt.action_id
        assert again.answer_receipt.run_id == run.id
    async with sessions() as fresh:
        assert (await fresh.get(OrchestrationAction, accepted.answer_receipt.action_id)).run_id == run.id
        assert await fresh.scalar(select(func.count()).select_from(EventLog).where(
            EventLog.dedup_key == f"authority.decision_resolved:{decision.id}")) == 1


async def _spr231_committed_runless(sessions, tmp_path):
    from tests.test_orchestration_roster_mutation import seed_project_user
    async with sessions() as db:
        project, user = await seed_project_user(db, tmp_path)
        goal = OrchestrationGoal(project_id=project.id, objective="Legacy ordinary no-work answer")
        db.add(goal)
        await db.flush()
        decision = await OrchestrationAuthorityDecisionService().create_pending(db, goal.id,
            decision_key="legacy:ordinary", title="Choose", question="Approve?", authority="human",
            options=["approve", "request_changes"])
        await db.commit()
        return project, user, goal, decision


@pytest.mark.parametrize("selected_option", ["approve", "request_changes"])
async def test_spr231_runless_saved_receipt_is_applied_readable_and_exact_replay_is_identity_bound(sessions, tmp_path, selected_option):
    from httpx import ASGITransport, AsyncClient
    from huddleroom.main import create_app
    from huddleroom.database import get_db
    from huddleroom.dependencies import get_current_user
    project, user, goal, decision = await _spr231_committed_runless(sessions, tmp_path)
    data = OrchestrationDecisionAnswerRequest(selected_option=selected_option, reason="Exact legacy reviewed answer")
    async with sessions() as submitting:
        accepted = await application.answer_decision(submitting, project.id, goal.id, decision.id, data, user)
        receipt = accepted.answer_receipt
        assert receipt.action_id is None and receipt.run_id is None and receipt.application_state == "applied"
        assert receipt.decision_ids == [decision.id] and receipt.actor_user_id == user.id
    app = create_app()
    async def request_db():
        async with sessions() as db:
            yield db
            await db.commit()
    async def actor():
        return user
    app.dependency_overrides[get_db], app.dependency_overrides[get_current_user] = request_db, actor
    base = f"/api/v1/projects/{project.id}/orchestration/goals/{goal.id}/decisions"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        readback = await client.get(f"{base}/{decision.id}")
        assert readback.status_code == 200, readback.text
        assert readback.json()["answer_receipt"] == receipt.model_dump(mode="json")
        assert (await client.get(base, params={"status": "pending"})).json() == []
    async with sessions() as replay:
        again = await application.answer_decision(replay, project.id, goal.id, decision.id, data, user)
        assert again.answer_receipt.model_dump() == receipt.model_dump()
    async with sessions() as setup:
        other = User(email=f"legacy-other-{uuid.uuid4()}@test.invalid", hashed_password="test")
        setup.add(other)
        await setup.commit()
    for changed, who in [
        (OrchestrationDecisionAnswerRequest(selected_option="approve" if selected_option == "request_changes" else "request_changes", reason=data.reason), user),
        (OrchestrationDecisionAnswerRequest(selected_option=selected_option, reason="Changed legacy reason"), user),
        (OrchestrationDecisionAnswerRequest(selected_option=selected_option, reason=data.reason, contract_version="changed"), user),
        (data, other),
    ]:
        async with sessions() as conflicting:
            with pytest.raises(HTTPException) as error:
                await application.answer_decision(conflicting, project.id, goal.id, decision.id, changed, who)
            assert error.value.status_code == 409
    async with sessions() as fresh:
        from huddleroom.models.orchestration import OrchestrationRun
        assert await fresh.scalar(select(func.count()).select_from(OrchestrationRun).where(OrchestrationRun.goal_id == goal.id)) == 0
        assert await fresh.scalar(select(func.count()).select_from(OrchestrationAction)) == 0
        assert await fresh.scalar(select(func.count()).select_from(EventLog).where(
            EventLog.dedup_key == f"authority.decision_resolved:{decision.id}")) == 1
        saved = await fresh.get(OrchestrationAuthorityDecision, decision.id)
        assert (saved.selected_option, saved.reason, saved.decided_by_user_id) == (selected_option, data.reason, user.id)


@pytest.mark.parametrize("explicit_caller", [False, True])
async def test_spr231_runless_completion_callback_failure_is_atomic(sessions, tmp_path, explicit_caller):
    from sqlalchemy import update
    from huddleroom.models.project import Project
    project, user, goal, decision = await _spr231_committed_runless(sessions, tmp_path)
    async def completion(db, response):
        assert response.decision.status == "answered" and response.answer_receipt.application_state == "applied"
        raise RuntimeError("Legacy receipt callback failed")
    async with sessions() as submitting:
        root = None
        if explicit_caller:
            await submitting.begin()
            await submitting.execute(update(Project).where(Project.id == project.id).values(description="Legacy unrelated write"))
            root = submitting.sync_session.get_transaction()
        with pytest.raises(RuntimeError, match="Legacy receipt callback"):
            await application.answer_decision(submitting, project.id, goal.id, decision.id,
                OrchestrationDecisionAnswerRequest(selected_option="approve", reason="Legacy answer"), user,
                completion_callback=completion)
        if explicit_caller:
            assert submitting.sync_session.get_transaction() is root and root.is_active
            assert await submitting.scalar(select(Project.description).where(Project.id == project.id)) == "Legacy unrelated write"
            await submitting.commit()
    async with sessions() as fresh:
        saved = await fresh.get(OrchestrationAuthorityDecision, decision.id)
        assert saved.status == "pending" and saved.selected_option is None
        assert await fresh.scalar(select(func.count()).select_from(EventLog).where(
            EventLog.dedup_key == f"authority.decision_resolved:{decision.id}")) == 0
        assert await fresh.scalar(select(Project.description).where(Project.id == project.id)) == (
            "Legacy unrelated write" if explicit_caller else project.description)


@pytest.mark.parametrize("native_mismatch", ["pending", "actor", "option", "reason", "membership", "timestamp"])
async def test_spr231_pending_runless_context_marker_cannot_forge_saved_readback_or_replay(sessions, tmp_path, native_mismatch):
    from httpx import ASGITransport, AsyncClient
    from huddleroom.main import create_app
    from huddleroom.database import get_db
    from huddleroom.dependencies import get_current_user
    project, user, goal, decision = await _spr231_committed_runless(sessions, tmp_path)
    data = OrchestrationDecisionAnswerRequest(selected_option="approve", reason="Actual native answer required")
    async with sessions() as setup:
        await application.answer_decision(setup, project.id, goal.id, decision.id, data, user)
    async with sessions() as forge:
        pending = await forge.get(OrchestrationAuthorityDecision, decision.id)
        # Preserve a fully well-shaped context record, but no actual native
        # answer exists. Metadata alone must never manufacture Saved authority.
        if native_mismatch == "pending":
            pending.status, pending.selected_option, pending.reason = "pending", None, None
            pending.decided_by_user_id, pending.decided_at = None, None
        elif native_mismatch == "actor":
            other = User(email=f"native-other-{uuid.uuid4()}@test.invalid", hashed_password="test")
            forge.add(other)
            await forge.flush()
            pending.decided_by_user_id = other.id
        elif native_mismatch == "option":
            pending.selected_option = "request_changes"
        elif native_mismatch == "reason":
            pending.reason = "Different native reason"
        elif native_mismatch == "membership":
            context = json.loads(pending.context)
            context["_ordinary_answer_acceptance"]["submission"]["decision_ids"].append(str(uuid.uuid4()))
            pending.context = json.dumps(context)
        else:
            pending.decided_at = None
        before = (pending.status, pending.selected_option, pending.reason, pending.decided_by_user_id)
        await forge.commit()
    app = create_app()
    async def request_db():
        async with sessions() as db:
            yield db
    async def actor():
        return user
    app.dependency_overrides[get_db], app.dependency_overrides[get_current_user] = request_db, actor
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        readback = await client.get(f"/api/v1/projects/{project.id}/orchestration/goals/{goal.id}/decisions/{decision.id}")
        assert readback.status_code in {200, 409}, readback.text
        if readback.status_code == 200:
            assert readback.json()["answer_receipt"] is None
    async def validation(db):
        raise RuntimeError("pending marker requires real validation")
    async with sessions() as submitting:
        try:
            await application.answer_decision(submitting, project.id, goal.id, decision.id, data, user,
                validation_callback=validation)
        except HTTPException as error:
            assert error.status_code == 409
        except RuntimeError as error:
            assert str(error) == "pending marker requires real validation"
        else:
            pytest.fail("pending context metadata replayed a forged Saved answer")
    async with sessions() as fresh:
        saved = await fresh.get(OrchestrationAuthorityDecision, decision.id)
        assert (saved.status, saved.selected_option, saved.reason, saved.decided_by_user_id) == before


async def test_spr231_public_accept_rejects_actor_mismatch_before_saved_replay(sessions, tmp_path):
    project, user, goal, decision = await _spr231_committed_runless(sessions, tmp_path)
    data = OrchestrationDecisionAnswerRequest(selected_option="approve", reason="Actor-bound native receipt")
    async with sessions() as setup:
        await application.answer_decision(setup, project.id, goal.id, decision.id, data, user)
    async with sessions() as setup:
        other = User(email=f"actor-mismatch-{uuid.uuid4()}@test.invalid", hashed_password="test")
        setup.add(other)
        await setup.commit()
    submission = application.canonical_answer_submission([decision], data, user.id)
    async with sessions() as replay:
        with pytest.raises(HTTPException) as error:
            await application.accept_answer(replay, project.id, goal.id, submission, other)
        assert error.value.status_code == 409
    async with sessions() as fresh:
        saved = await fresh.get(OrchestrationAuthorityDecision, decision.id)
        assert saved.decided_by_user_id == user.id and saved.reason == data.reason


@pytest.mark.parametrize("malformed", ["single_multiple", "single_duplicate", "batch_duplicate"])
async def test_spr231_exact_accept_rejects_invalid_cardinality_before_callbacks_or_writes(sessions, tmp_path, malformed):
    project, user, goal, first = await _spr231_committed_runless(sessions, tmp_path)
    async with sessions() as setup:
        second = await OrchestrationAuthorityDecisionService().create_pending(setup, goal.id,
            decision_key="legacy:second", title="Second", question="Second approval?", authority="human",
            options=["approve", "request_changes"])
        await setup.commit()
    data = OrchestrationDecisionAnswerRequest(selected_option="approve", reason="Each native answer requires exact cardinality")
    left = application.canonical_answer_submission([first], data, user.id)
    right = application.canonical_answer_submission([second], data, user.id)
    other = right if malformed == "single_multiple" else left
    submission = left.model_copy(update={
        "kind": "batch" if malformed == "batch_duplicate" else "single",
        "decision_ids": [first.id, second.id if malformed == "single_multiple" else first.id],
        "answers": [left.answers[0], other.answers[0]], "contracts": [left.contracts[0], other.contracts[0]],
    })
    callbacks = []
    async def validation(db):
        callbacks.append("validation")
    async def reservation(db, linkage):
        callbacks.append("reservation")
    async def completion(db, response):
        callbacks.append("completion")
    async with sessions() as submitting:
        with pytest.raises(HTTPException) as error:
            await application.accept_answer(submitting, project.id, goal.id, submission, user,
                validation_callback=validation, reservation_callback=reservation, completion_callback=completion)
        assert error.value.status_code == 409
    assert callbacks == []
    async with sessions() as fresh:
        for decision_id in [first.id, second.id]:
            row = await fresh.get(OrchestrationAuthorityDecision, decision_id)
            assert row.status == "pending" and row.selected_option is None and row.continuation_action_id is None
            assert await fresh.scalar(select(func.count()).select_from(EventLog).where(
                EventLog.dedup_key == f"authority.decision_resolved:{decision_id}")) == 0
        assert await fresh.scalar(select(func.count()).select_from(OrchestrationAction)) == 0
