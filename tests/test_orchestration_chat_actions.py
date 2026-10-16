"""Finite chat actions retain durable review, frozen targets and real domain gates."""
import uuid
import asyncio
import json
from contextlib import asynccontextmanager

import pytest
from fastapi import HTTPException
from sqlalchemy import select, func

from huddleroom.config import settings
from huddleroom.models import User
from huddleroom.models.event_log import EventLog
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGoal, OrchestrationRun
from huddleroom.models.orchestration_chat import ProjectChatTurn
from huddleroom.schemas.orchestration_chat import ChatActionDraft, ChatInterpretation
from tests.test_orchestration_roster_mutation import seed_project_user, seed_goal_run, sessions
from tests.test_orchestration_roster_mutation import (
    approve_semantic_reviews, assert_entry_handoff, lock_tracer, seed_definition_proposal,
    seed_hierarchy_proposal, single_agent_write, fresh_agent, PROPOSED_DESCRIPTION,
)
from tests.test_orchestration_recovery_authority import fail_on_effect

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("safe_goal_analysis")]


def _service(sessions, **kwargs):
    from huddleroom.services.orchestration_chat_actions import OrchestrationChatActionService
    return OrchestrationChatActionService(session_factory=sessions, **kwargs)


async def _seed(sessions, tmp_path, *, status="active", phase="baseline"):
    async with sessions() as db:
        project, user = await seed_project_user(db, tmp_path)
        goal, run = await seed_goal_run(db, project, user=user)
        goal.status, run.status = status, "paused" if status == "paused" else "running"
        run.phase = phase
        await db.commit()
        return project, user, goal, run


async def _prepare(sessions, user, goal, operation, *, content=None, target_id=None, payload=None, named_goal=None):
    service = _service(sessions)
    async with sessions() as db:
        turn = ProjectChatTurn(project_id=goal.project_id, actor_id=user.id,
            client_request_id=uuid.uuid4(), content=content or f"{operation.replace('_', ' ')} this goal",
            context_goal_id=goal.id, source_kind="action")
        db.add(turn)
        await db.flush()
        action = ChatActionDraft(action_id=uuid.uuid4(), operation=operation,
            project_id=goal.project_id, goal_id=goal.id, target_id=target_id,
            payload=payload or {}, summary="Reviewed action")
        turn.interpretation = ChatInterpretation(intent="instruction", operation=operation,
            goal_id=goal.id, named_goal=named_goal, payload=action.payload).model_dump(mode="json")
        prepared = await service.prepare(db, turn, action)
        await db.commit()
        return turn, prepared


async def _states(sessions, goal, run):
    async with sessions() as db:
        return await db.get(OrchestrationGoal, goal.id), await db.get(OrchestrationRun, run.id)


@pytest.mark.parametrize("operation,terminal", [("pause", "paused"), ("cancel", "cancelled")])
async def test_mandatory_confirmation_and_duplicate_receipt_converge(sessions, tmp_path, operation, terminal, monkeypatch):
    project, user, goal, run = await _seed(sessions, tmp_path)
    turn, action = await _prepare(sessions, user, goal, operation)
    assert action.requires_confirmation
    service = _service(sessions)
    async with sessions() as db:
        pending = await service.execute(db, await db.get(ProjectChatTurn, turn.id), action)
        await db.commit()
    assert pending.status == "pending"
    stored_goal, stored_run = await _states(sessions, goal, run)
    assert stored_goal.status == "active" and stored_run.status == "running"
    monkeypatch.setattr(settings, "orchestration_advisor_allowance_tokens", 0)
    monkeypatch.setattr(settings, "orchestration_conversation_allowance_tokens", 0)
    request_id = uuid.uuid4()
    first = await service.confirm(project.id, user.id, action.action_id, request_id)
    second = await service.confirm(project.id, user.id, action.action_id, request_id)
    assert first == second
    stored_goal, stored_run = await _states(sessions, goal, run)
    assert stored_goal.status == stored_run.status == terminal
    if operation == "pause":
        assert stored_run.baseline_authorized is False
        assert stored_run.pause_requested is True and stored_run.control_version == 1
        async with sessions() as fresh:
            controls = (await fresh.scalars(select(OrchestrationAction).where(
                OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == "pause_control"))).all()
            assert len(controls) == 1
    async with sessions() as db:
        row = await db.get(ProjectChatTurn, turn.id)
        assert row.action_receipt["action_id"] == str(action.action_id)


async def test_resume_is_direct_and_does_not_restore_baseline_grant(sessions, tmp_path):
    _, user, goal, run = await _seed(sessions, tmp_path, status="paused")
    async with sessions() as db:
        stored_run = await db.get(OrchestrationRun, run.id)
        stored_run.baseline_authorized = False
        await db.commit()
    turn, action = await _prepare(sessions, user, goal, "resume")
    assert action.requires_confirmation is False
    async with sessions() as db:
        receipt = await _service(sessions).execute(db, await db.get(ProjectChatTurn, turn.id), action)
        await db.commit()
    assert receipt.status in {"applied", "completed"}
    stored_goal, stored_run = await _states(sessions, goal, run)
    assert stored_goal.status == "active" and stored_run.status == "running"
    assert stored_run.baseline_authorized is False


async def test_frozen_confirmation_rejects_drift(sessions, tmp_path):
    project, user, goal, run = await _seed(sessions, tmp_path)
    _, action = await _prepare(sessions, user, goal, "pause")
    async with sessions() as db:
        old = await db.get(OrchestrationRun, run.id)
        old.status = "cancelled"
        replacement = OrchestrationRun(goal_id=goal.id, status="running", phase="baseline")
        db.add(replacement)
        await db.commit()
    with pytest.raises(HTTPException) as error:
        await _service(sessions).confirm(project.id, user.id, action.action_id, uuid.uuid4())
    assert error.value.status_code == 409
    stored_goal, _ = await _states(sessions, goal, run)
    assert stored_goal.status == "active"
    async with sessions() as db:
        assert (await db.get(OrchestrationRun, replacement.id)).status == "running"


@pytest.mark.parametrize("foreign", ["actor", "project", "action"])
async def test_confirmation_rejects_foreign_references(sessions, tmp_path, foreign):
    project, user, goal, run = await _seed(sessions, tmp_path)
    _, action = await _prepare(sessions, user, goal, "pause")
    with pytest.raises(HTTPException) as error:
        await _service(sessions).confirm(uuid.uuid4() if foreign == "project" else project.id,
            uuid.uuid4() if foreign == "actor" else user.id,
            uuid.uuid4() if foreign == "action" else action.action_id, uuid.uuid4())
    assert error.value.status_code in {403, 404, 409}
    stored_goal, stored_run = await _states(sessions, goal, run)
    assert stored_goal.status == "active" and stored_run.status == "running"


@pytest.mark.parametrize("content,operation", [("Why is Alpha paused?", "resume"),
    ("Prioritize verification", "resume"), ("Start this goal", "start_execution")])
async def test_action_preparation_rejects_missing_or_ambiguous_instruction_evidence(sessions, tmp_path, content, operation):
    _, user, goal, run = await _seed(sessions, tmp_path)
    with pytest.raises(HTTPException) as error:
        await _prepare(sessions, user, goal, operation, content=content)
    assert error.value.status_code == 409
    stored_goal, stored_run = await _states(sessions, goal, run)
    assert stored_goal.status == "active" and stored_run.phase == "baseline"


async def test_model_supplied_goal_cannot_override_captured_unnamed_goal(sessions, tmp_path):
    project, user, goal, _ = await _seed(sessions, tmp_path)
    async with sessions() as db:
        other, _ = await seed_goal_run(db, project, user=user)
        turn = ProjectChatTurn(project_id=project.id, actor_id=user.id, client_request_id=uuid.uuid4(),
                               content="Pause this goal", context_goal_id=goal.id)
        db.add(turn)
        await db.flush()
        draft = ChatActionDraft(action_id=uuid.uuid4(), operation="pause", project_id=project.id,
                                goal_id=other.id, summary="Pause other")
        turn.interpretation = ChatInterpretation(intent="instruction", operation="pause",
            goal_id=other.id).model_dump(mode="json")
        with pytest.raises(HTTPException) as error:
            await _service(sessions).prepare(db, turn, draft)
        assert error.value.status_code == 409


async def _ready(sessions, tmp_path):
    from tests.test_orchestration_authorization import _seed_baseline_terminal_run
    async with sessions() as db:
        project, user = await seed_project_user(db, tmp_path)
        goal, run = await _seed_baseline_terminal_run(db, project)
        goal.status, run.status, run.phase = "active", "running", "ready"
        await db.commit()
        return project, user, goal, run


async def test_execution_start_keeps_confirmation_and_one_authorization(sessions, tmp_path):
    project, user, goal, run = await _ready(sessions, tmp_path)
    turn, action = await _prepare(sessions, user, goal, "start_execution", content="Start execution work for this goal")
    assert action.requires_confirmation
    async with sessions() as db:
        pending = await _service(sessions).execute(db, await db.get(ProjectChatTurn, turn.id), action)
        await db.commit()
    assert pending.status == "pending"
    assert (await _states(sessions, goal, run))[1].phase == "ready"
    service = _service(sessions)
    request_id = uuid.uuid4()
    await service.confirm(project.id, user.id, action.action_id, request_id)
    await service.confirm(project.id, user.id, action.action_id, request_id)
    assert (await _states(sessions, goal, run))[1].phase == "authorized"
    async with sessions() as db:
        actions = (await db.scalars(select(OrchestrationAction).where(
            OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == "authorize_execution"))).all()
        assert len(actions) == 1


async def test_start_envelope_drift_requires_refreshed_review_and_new_acceptance(sessions, tmp_path):
    project, user, goal, run = await _ready(sessions, tmp_path)
    turn, action = await _prepare(sessions, user, goal, "start_execution", content="Start execution work for this goal")
    async with sessions() as db:
        stored = await db.get(OrchestrationGoal, goal.id)
        stored.budget = {"max_tokens": 200}
        await db.commit()
    stale = await _service(sessions).confirm(project.id, user.id, action.action_id, uuid.uuid4())
    assert stale.status == "pending" and stale.error_code == "chat_stale_review"
    assert (await _states(sessions, goal, run))[1].phase == "ready"
    async with sessions() as db:
        row = await db.get(ProjectChatTurn, turn.id)
        refreshed = ChatActionDraft.model_validate(row.action_draft)
    assert refreshed.frozen_versions != action.frozen_versions
    assert refreshed.action_id != action.action_id
    assert "200" in refreshed.summary
    # Explicitly accepting the newly shown summary is a fresh no-LLM confirmation.
    result = await _service(sessions).confirm(project.id, user.id, refreshed.action_id, uuid.uuid4())
    assert result.status == "completed"
    assert (await _states(sessions, goal, run))[1].phase == "authorized"


async def test_baseline_authorization_never_chains_execution(sessions, tmp_path):
    _, user, goal, run = await _seed(sessions, tmp_path)
    async with sessions() as db:
        stored = await db.get(OrchestrationRun, run.id)
        stored.baseline_authorized = False
        await db.commit()
    turn, action = await _prepare(sessions, user, goal, "start_baseline", content="Start baseline for this goal")
    async with sessions() as db:
        result = await _service(sessions).execute(db, await db.get(ProjectChatTurn, turn.id), action)
        await db.commit()
    assert result.status == "completed"
    stored = (await _states(sessions, goal, run))[1]
    assert stored.baseline_authorized is True and stored.phase == "baseline"
    async with sessions() as db:
        assert (await db.scalars(select(OrchestrationAction).where(
            OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == "authorize_execution"))).all() == []


async def test_reviewed_payload_cannot_be_replaced_before_confirmation(sessions, tmp_path):
    _, user, goal, run = await _seed(sessions, tmp_path)
    turn, action = await _prepare(sessions, user, goal, "pause")
    forged = action.model_copy(update={"requires_confirmation": False})
    async with sessions() as db:
        with pytest.raises(HTTPException) as error:
            await _service(sessions).execute(db, await db.get(ProjectChatTurn, turn.id), forged)
        assert error.value.status_code == 409
    assert (await _states(sessions, goal, run))[0].status == "active"


async def test_pause_effect_and_receipt_rollback_together_before_commit(sessions, tmp_path):
    from huddleroom.services.orchestration_service import OrchestrationService
    project, user, goal, run = await _seed(sessions, tmp_path)
    turn, action = await _prepare(sessions, user, goal, "pause")

    class FailedPublication(OrchestrationService):
        async def pause_goal(self, *args, **kwargs):
            await super().pause_goal(*args, **kwargs)
            raise RuntimeError("crash after transactional effect before receipt")

    with pytest.raises(RuntimeError):
        await _service(sessions, orchestration=FailedPublication()).confirm(project.id, user.id, action.action_id, uuid.uuid4())
    stored_goal, stored_run = await _states(sessions, goal, run)
    assert stored_goal.status == "active" and stored_run.status == "running"
    assert stored_run.pause_requested is False and stored_run.control_version == 0
    async with sessions() as db:
        assert (await db.scalars(select(OrchestrationAction).where(
            OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == "pause_control"))).all() == []
        assert (await db.get(ProjectChatTurn, turn.id)).action_receipt is None
    # Verified rollback permits the same prepared request to execute once.
    await _service(sessions).confirm(project.id, user.id, action.action_id, uuid.uuid4())
    assert (await _states(sessions, goal, run))[0].status == "paused"


async def _decision(sessions, goal, run, *, key="chat-human", options=None):
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
    async with sessions() as db:
        decision = await OrchestrationAuthorityDecisionService().create_pending(db, goal.id,
            decision_key=key, title="Choose", question="Proceed?", authority="human", run_id=run.id,
            options=options or ["approve", "reject"])
        await db.commit()
        return decision


async def test_single_decision_option_cannot_be_taken_from_explanatory_text(sessions, tmp_path):
    _, user, goal, run = await _seed(sessions, tmp_path)
    decision = await _decision(sessions, goal, run)
    with pytest.raises(HTTPException):
        await _prepare(sessions, user, goal, "answer_authority", target_id=decision.id,
            content=f"Reject {decision.id} because approve is unsafe", payload={"selected_option": "approve"})


async def test_swapped_batch_answers_cannot_borrow_other_decision_option(sessions, tmp_path):
    _, user, goal, run = await _seed(sessions, tmp_path)
    first = await _decision(sessions, goal, run, key="agent_definition_review:proposal:first")
    second = await _decision(sessions, goal, run, key="agent_definition_review:proposal:second")
    with pytest.raises(HTTPException):
        await _prepare(sessions, user, goal, "answer_agent_definition_batch",
            content=f"Approve {first.id}; reject {second.id}", payload={"answers": [
                {"decision_id": str(first.id), "selected_option": "reject"},
                {"decision_id": str(second.id), "selected_option": "approve"}]})


@pytest.mark.parametrize("payload", [{"selected_option": "reject", "reason": "Invented explanation"},
    {"selected_option": "edit", "edited_description": "Invented description", "edited_persona": "Invented persona"}])
async def test_required_reason_and_edits_cannot_be_invented_by_provider(sessions, tmp_path, payload):
    _, user, goal, run = await _seed(sessions, tmp_path)
    decision = await _decision(sessions, goal, run, options=["approve", "reject", "edit"])
    with pytest.raises(HTTPException):
        await _prepare(sessions, user, goal, "answer_authority", target_id=decision.id,
            content=f"Answer {decision.id} {payload['selected_option']}", payload=payload)


@pytest.mark.parametrize("natural", [False, True])
async def test_human_answer_uses_specialized_application_and_durable_receipt(sessions, tmp_path, natural):
    _, user, goal, run = await _seed(sessions, tmp_path)
    decision = await _decision(sessions, goal, run)
    turn, action = await _prepare(sessions, user, goal, "answer_authority", target_id=decision.id,
        content=f"Approve decision {decision.id}" if natural else f"Answer {decision.id} approve",
        payload={"selected_option": "approve"})
    async with sessions() as db:
        receipt = await _service(sessions).execute(db, await db.get(ProjectChatTurn, turn.id), action)
        await db.commit()
    assert receipt.status == "recorded" and receipt.source_record_id == decision.id
    from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
    async with sessions() as db:
        assert (await db.get(OrchestrationAuthorityDecision, decision.id)).status == "answered"
        assert (await db.get(ProjectChatTurn, turn.id)).action_receipt["status"] == "recorded"


async def test_concurrent_confirmations_converge_one_effect_receipt(sessions, tmp_path):
    from huddleroom.services.orchestration_service import OrchestrationService
    project, user, goal, run = await _seed(sessions, tmp_path)
    turn, action = await _prepare(sessions, user, goal, "pause")
    calls = []

    class Counted(OrchestrationService):
        async def pause_goal(self, *args, **kwargs):
            calls.append(True)
            return await super().pause_goal(*args, **kwargs)

    service = _service(sessions, orchestration=Counted())
    client_request_id = uuid.uuid4()
    results = await asyncio.gather(*[service.confirm(project.id, user.id, action.action_id, client_request_id) for _ in range(2)])
    assert results[0] == results[1]
    assert len(calls) == 1
    assert (await _states(sessions, goal, run))[0].status == "paused"
    async with sessions() as db:
        assert (await db.get(ProjectChatTurn, turn.id)).action_receipt["action_id"] == str(action.action_id)


async def test_steering_paused_direction_is_pending_and_preserves_ownership(sessions, tmp_path, monkeypatch):
    from huddleroom.models.orchestration_steering import OrchestrationSteeringRequest
    monkeypatch.setattr(settings, "orchestration_conversation_steering_enabled", True)
    _, user, goal, run = await _seed(sessions, tmp_path, status="paused", phase="authorized")
    payload = {"directive": "Prioritize verification", "scope": "run", "lifetime": "remaining_current_run",
               "target_type": "goal", "target_id": str(goal.id), "impact_summary": "Focus the current run"}
    turn, action = await _prepare(sessions, user, goal, "submit_steering", content="Prioritize verification", payload=payload)
    async with sessions() as db:
        receipt = await _service(sessions).execute(db, await db.get(ProjectChatTurn, turn.id), action)
        await db.commit()
    assert receipt.status == "pending"
    async with sessions() as db:
        request = await db.get(OrchestrationSteeringRequest, receipt.source_record_id)
        assert request.status == "pending" and request.submitted_run_id == run.id
        assert request.actor_id == user.id


@pytest.mark.parametrize("altered", [False, True])
async def test_promotion_requires_complete_original_proposal_payload(sessions, tmp_path, monkeypatch, altered):
    from huddleroom.models.orchestration_conversation import ConversationMessage, ConversationResponse
    from huddleroom.models.orchestration_steering import OrchestrationSteeringProposal
    monkeypatch.setattr(settings, "orchestration_conversation_steering_enabled", True)
    _, user, goal, run = await _seed(sessions, tmp_path, phase="authorized")
    payload = {"directive": "Prioritize verification", "scope": "run", "lifetime": "remaining_current_run",
               "target_type": "goal", "target_id": str(goal.id), "impact_summary": "Focus the current run"}
    async with sessions() as db:
        message = ConversationMessage(goal_id=goal.id, actor_id=user.id, client_request_id=uuid.uuid4(), sequence=1, content="Advice")
        db.add(message)
        await db.flush()
        response = ConversationResponse(message_id=message.id, dossier={}, context_manifest={}, context_version="v1",
                                        provider_request_id=f"promotion:{message.id}")
        db.add(response)
        await db.flush()
        proposal = OrchestrationSteeringProposal(id=uuid.uuid4(), response_id=response.id,
                                                goal_id=goal.id, actor_id=user.id, draft=payload)
        db.add(proposal)
        await db.commit()
    changed = {**payload, "scope": "goal", "lifetime": "future_runs"} if altered else payload
    turn, action = await _prepare(sessions, user, goal, "promote_steering", content=f"Promote {proposal.id}",
                                  target_id=proposal.id, payload=changed)
    async with sessions() as db:
        if altered:
            with pytest.raises(HTTPException):
                await _service(sessions).execute(db, await db.get(ProjectChatTurn, turn.id), action)
            await db.rollback()
        else:
            receipt = await _service(sessions).execute(db, await db.get(ProjectChatTurn, turn.id), action)
            assert receipt.status == "pending"
            await db.commit()
    async with sessions() as db:
        assert (await db.get(OrchestrationSteeringProposal, proposal.id)).status == ("proposed" if altered else "promoted")


async def test_duplicate_objective_named_target_requires_clarification(sessions, tmp_path):
    project, user, goal, _ = await _seed(sessions, tmp_path)
    async with sessions() as db:
        other, _ = await seed_goal_run(db, project, user=user)
        stored = await db.get(OrchestrationGoal, goal.id)
        stored.objective = other.objective = "Duplicate objective"
        await db.commit()
    with pytest.raises(HTTPException):
        await _prepare(sessions, user, goal, "pause", content="Pause Duplicate objective", named_goal="Duplicate objective")


@pytest.mark.parametrize("denial", ["workspace", "continuous_policy"])
async def test_existing_start_denials_and_continuous_policy_gate_remain(sessions, tmp_path, denial):
    project, user, goal, run = await _ready(sessions, tmp_path)
    async with sessions() as db:
        if denial == "workspace":
            from huddleroom.models import Project
            (await db.get(Project, project.id)).workspace_path = None
        else:
            stored = await db.get(OrchestrationGoal, goal.id)
            stored.goal_type, stored.planning_mode = "continuous", "continuous_policy"
            stored.continuous_policy = None
        await db.commit()
    turn, action = await _prepare(sessions, user, goal, "start_execution", content="Start execution work for this goal")
    with pytest.raises(HTTPException):
        await _service(sessions).confirm(project.id, user.id, action.action_id, uuid.uuid4())
    assert (await _states(sessions, goal, run))[1].phase == "ready"
    async with sessions() as db:
        assert (await db.get(ProjectChatTurn, turn.id)).action_receipt is None


async def _loaded_goal_user(sessions, seeded):
    async with sessions() as db:
        return await db.get(OrchestrationGoal, seeded.goal_id), await db.get(User, seeded.user_id)


async def _drain_saved_answer(sessions, run_id):
    from huddleroom.services.orchestration_decision_application import drain_answer_applications
    async with sessions() as consumer:
        await drain_answer_applications(consumer, run_id)
        await consumer.commit()


async def test_agent_batch_production_handoff_releases_chat_goal_lock(sessions, tmp_path, approve_semantic_reviews, lock_tracer):
    seeded = await seed_definition_proposal(sessions, tmp_path)
    goal, user = await _loaded_goal_user(sessions, seeded)
    turn, action = await _prepare(sessions, user, goal, "answer_agent_definition_batch",
        content=f"Answer {seeded.decision_id} approve", payload={"answers": [
            {"decision_id": str(seeded.decision_id), "selected_option": "approve"}]})
    lock_tracer.start()
    async with sessions() as db:
        receipt = await _service(sessions).execute(db, await db.get(ProjectChatTurn, turn.id), action)
        await db.commit()
    assert receipt.status == "recorded"
    # Saved acknowledgment precedes the fresh native roster consumer.
    await _drain_saved_answer(sessions, seeded.run_id)
    assert_entry_handoff(lock_tracer, goal.id)
    single_agent_write(lock_tracer, agent_id=seeded.agent_id)
    assert (await fresh_agent(sessions, seeded.agent_id)).description == PROPOSED_DESCRIPTION
    async with sessions() as db:
        row = await db.get(ProjectChatTurn, turn.id)
        assert row.frozen_context["action_reservation"]


async def test_hierarchy_production_handoff_releases_chat_goal_lock(sessions, tmp_path, monkeypatch, lock_tracer):
    seeded = await seed_hierarchy_proposal(sessions, tmp_path, monkeypatch)
    goal, user = await _loaded_goal_user(sessions, seeded)
    turn, action = await _prepare(sessions, user, goal, "answer_hierarchy", target_id=seeded.decision_id,
        content=f"Answer {seeded.decision_id} approve", payload={"selected_option": "approve"})
    lock_tracer.start()
    async with sessions() as db:
        receipt = await _service(sessions).execute(db, await db.get(ProjectChatTurn, turn.id), action)
        await db.commit()
    assert receipt.status == "recorded"
    # Saved acknowledgment precedes the fresh native roster consumer.
    await _drain_saved_answer(sessions, seeded.run_id)
    assert_entry_handoff(lock_tracer, goal.id)
    single_agent_write(lock_tracer, name=seeded.proposal_name)


@pytest.mark.parametrize("case", ["pretty", "reordered", "quoted_command", "question", "hypothetical", "mismatch", "trailing_command"])
async def test_hierarchy_typed_edit_interpretation_to_production_action(sessions, tmp_path, monkeypatch, case):
    from huddleroom.services.orchestration_chat_interpretation import parse_chat_interpretation
    from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
    from tests.test_orchestration_roster_mutation import proposed_agent_definition, agents_named

    seeded = await seed_hierarchy_proposal(sessions, tmp_path, monkeypatch)
    goal, user = await _loaded_goal_user(sessions, seeded)
    edit = {**proposed_agent_definition()["definition"], "name": f"edited-{seeded.proposal_name}"}
    # An instruction-looking string inside typed agent data cannot authorize a lifecycle effect.
    edit["system_prompt"] = "Pause this goal"
    human_edit = dict(reversed(list(edit.items()))) if case == "reordered" else dict(edit)
    if case == "mismatch":
        human_edit["name"] = "different-human-name"
    prefix = f"Answer decision {seeded.decision_id} edit"
    if case == "quoted_command":
        prefix = f'"{prefix}"'
    elif case == "question":
        prefix = f"Should I answer decision {seeded.decision_id} edit?"
    elif case == "hypothetical":
        prefix = f"If I answer decision {seeded.decision_id} edit"
    content = prefix + "\nedited_agent: " + json.dumps(human_edit, indent=2)
    if case == "trailing_command":
        content += "\nPause this goal"
    parsed = parse_chat_interpretation(json.dumps({"intent": "instruction", "operation": "answer_hierarchy",
        "goal_id": str(goal.id),
        "payload": {"selected_option": "edit", "edited_agent": edit}}), content, goal.id)
    valid = case in {"pretty", "reordered"}
    if valid:
        assert parsed.intent == "instruction" and parsed.payload.edited_agent.name == edit["name"]
        async with sessions() as db:
            turn = ProjectChatTurn(project_id=goal.project_id, actor_id=user.id, client_request_id=uuid.uuid4(),
                content=content, context_goal_id=goal.id, interpretation=parsed.model_dump(mode="json"))
            db.add(turn)
            await db.flush()
            action = ChatActionDraft(action_id=uuid.uuid4(), operation=parsed.operation, project_id=goal.project_id,
                goal_id=goal.id, target_id=seeded.decision_id, payload=parsed.payload, summary="Edit hierarchy agent")
            action = await _service(sessions).prepare(db, turn, action)
            await db.commit()
            receipt = await _service(sessions).execute(db, turn, action)
            await db.commit()
            assert receipt.status == "recorded"
        await _drain_saved_answer(sessions, seeded.run_id)
    else:
        assert parsed.intent != "instruction"
    async with sessions() as db:
        decision = await db.get(OrchestrationAuthorityDecision, seeded.decision_id)
        assert decision.status == ("answered" if valid else "pending")
        assert len(await agents_named(sessions, edit["name"])) == (1 if valid else 0)
        assert (await db.get(OrchestrationGoal, goal.id)).status != "paused"


async def test_roster_reservation_crash_is_linked_and_never_redispatched(
    sessions, tmp_path, approve_semantic_reviews, monkeypatch
):
    import huddleroom.services.orchestration_decision_application as application
    seeded = await seed_definition_proposal(sessions, tmp_path)
    goal, user = await _loaded_goal_user(sessions, seeded)
    turn, action = await _prepare(sessions, user, goal, "answer_agent_definition_batch",
        content=f"Answer {seeded.decision_id} approve", payload={"answers": [
            {"decision_id": str(seeded.decision_id), "selected_option": "approve"}]})
    handoffs = []
    class ProcessTerminated(BaseException):
        pass

    async def crash(*_args, **_kwargs):
        handoffs.append(True)
        raise ProcessTerminated("crash after reservation before roster handoff")

    from huddleroom.services import agent_service
    original_handoff = agent_service.complete_roster_handoff
    # Acceptance is already durable; interrupt the real consumer at its native
    # published roster reservation, before roster application/handoff.
    async with sessions() as db:
        saved = await _service(sessions).execute(db, await db.get(ProjectChatTurn, turn.id), action)
        assert saved.status == "recorded"
    monkeypatch.setattr(agent_service, "complete_roster_handoff", crash)
    with pytest.raises(ProcessTerminated, match="crash after reservation"):
        await _drain_saved_answer(sessions, seeded.run_id)
    async with sessions() as db:
        row = await db.get(ProjectChatTurn, turn.id)
        assert row.action_receipt["status"] == "recorded"
        assert row.frozen_context["action_reservation"]
        replay = await _service(sessions).execute(db, row, action)
        await db.commit()
    assert replay.status == "recorded"
    assert len(handoffs) == 1
    monkeypatch.setattr(agent_service, "complete_roster_handoff", original_handoff)
    await _drain_saved_answer(sessions, seeded.run_id)
    await _drain_saved_answer(sessions, seeded.run_id)
    assert (await fresh_agent(sessions, seeded.agent_id)).description == PROPOSED_DESCRIPTION



@pytest.mark.parametrize("operation", ["pause", "cancel", "start_execution"])
async def test_owned_effect_and_receipt_commit_before_goal_lock_releases(sessions, tmp_path, operation):
    from huddleroom.services.orchestration_service import OrchestrationService
    project, user, goal, _ = await (_ready(sessions, tmp_path) if operation == "start_execution" else _seed(sessions, tmp_path))
    turn, action = await _prepare(sessions, user, goal, operation,
        content="Start execution work for this goal" if operation == "start_execution" else None)
    observed = []

    class ObservedLock(OrchestrationService):
        depth = 0

        @asynccontextmanager
        async def _lock_goal_for_baseline_transition(self, db, goal_id):
            async with super()._lock_goal_for_baseline_transition(db, goal_id):
                self.depth += 1
                try:
                    yield
                finally:
                    if self.depth == 1:
                        # Production-like independent reader sees both committed records before
                        # the originating goal lock is released, including services that commit internally.
                        async with sessions() as fresh:
                            row = await fresh.get(ProjectChatTurn, turn.id)
                            stored_goal = await fresh.get(OrchestrationGoal, goal.id)
                            assert row.action_receipt is not None
                            assert row.action_receipt["status"] == "completed"
                            assert row.action_receipt["action_id"] == str(action.action_id)
                            if operation in {"pause", "cancel"}:
                                assert stored_goal.status == ("paused" if operation == "pause" else "cancelled")
                            observed.append(True)
                    self.depth -= 1

    await _service(sessions, orchestration=ObservedLock()).confirm(project.id, user.id, action.action_id, uuid.uuid4())
    if operation == "pause":
        # SPR230/SPR231 Pause deliberately bypasses the long transition lock.
        assert observed == []
        async with sessions() as fresh:
            row = await fresh.get(ProjectChatTurn, turn.id)
            stored_goal = await fresh.get(OrchestrationGoal, goal.id)
            assert row.action_receipt["action_id"] == str(action.action_id)
            assert stored_goal.status == "paused"
            controls = (await fresh.scalars(select(OrchestrationAction).where(
                OrchestrationAction.action_type == "pause_control"))).all()
            assert len(controls) == 1
    else:
        assert observed == [True]


@pytest.mark.parametrize("operation", ["answer_runtime_question", "answer_authority"])
async def test_runtime_question_requires_specialized_path_and_exactly_one_continuation(sessions, tmp_path, operation):
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
    _, user, goal, run = await _seed(sessions, tmp_path)
    async with sessions() as db:
        decision = await OrchestrationAuthorityDecisionService().create_runtime_question(db, goal.id,
            run_id=run.id, subject="release", authority="human", contract_version="plan:1",
            continuation={"action_type": "continue", "reason": "Approved"},
            question="Proceed?", options=[{"key": "approve"}, {"key": "reject"}])
        await db.commit()
    turn, action = await _prepare(sessions, user, goal, operation, target_id=decision.id,
        content=f"Answer {decision.id} approve", payload={"selected_option": "approve", "contract_version": "plan:1"})
    async with sessions() as db:
        if operation == "answer_authority":
            with pytest.raises(HTTPException):
                await _service(sessions).execute(db, await db.get(ProjectChatTurn, turn.id), action)
            await db.rollback()
        else:
            first = await _service(sessions).execute(db, await db.get(ProjectChatTurn, turn.id), action)
            await db.commit()
            second = await _service(sessions).execute(db, await db.get(ProjectChatTurn, turn.id), action)
            assert first == second and first.status == "recorded"
            await db.commit()
    async with sessions() as db:
        continuations = (await db.scalars(select(OrchestrationAction).where(
            OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == "decision_continuation"))).all()
        assert len(continuations) == (0 if operation == "answer_authority" else 1)


async def test_recovery_answer_links_one_reserved_wrapper_without_effects(sessions, tmp_path, fail_on_effect):
    from tests.test_orchestration_recovery_authority import diagnosed, actions_of
    from tests.test_orchestration_roster_mutation import make_agent
    _, user, goal, run = await _seed(sessions, tmp_path)
    async with sessions() as db:
        agent = make_agent(name=f"planner-{uuid.uuid4()}", capabilities=["planning"])
        db.add(agent)
        await db.flush()
        _, _, proposal, decision = await diagnosed(db, await db.get(OrchestrationGoal, goal.id),
            await db.get(OrchestrationRun, run.id), agent, owns_setup=True)
        await db.commit()
    turn, action = await _prepare(sessions, user, goal, "approve_recovery", target_id=decision.id,
        content=f"Approve {decision.id} approve_fix", payload={"selected_option": "approve_fix"})
    async with sessions() as db:
        first = await _service(sessions).execute(db, await db.get(ProjectChatTurn, turn.id), action)
        await db.commit()
        replay = await _service(sessions).execute(db, await db.get(ProjectChatTurn, turn.id), action)
        assert first == replay and first.status == "recorded"
        await db.commit()
    async with sessions() as db:
        wrappers = await actions_of(db, run, "recovery_apply")
        assert len(wrappers) == 1
        linkage = (await db.get(ProjectChatTurn, turn.id)).frozen_context["action_reservation"]
        assert str(proposal.id) in str(linkage)
        assert str(wrappers[0].id) in str(linkage)


async def test_natural_approval_with_multiple_matching_offered_aliases_clarifies(sessions, tmp_path):
    _, user, goal, run = await _seed(sessions, tmp_path)
    decision = await _decision(sessions, goal, run, options=["approve", "accept", "reject"])
    with pytest.raises(HTTPException):
        await _prepare(sessions, user, goal, "answer_authority", target_id=decision.id,
                       content=f"Approve decision {decision.id}", payload={"selected_option": "approve"})


@pytest.mark.parametrize("runtime", [False, True, "batch"])
@pytest.mark.parametrize("failure", [False, True])
async def test_decision_receipt_publishes_inside_native_lock_or_rolls_back(sessions, tmp_path, monkeypatch, runtime, failure):
    from huddleroom.services.orchestration_service import OrchestrationService
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
    from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
    import huddleroom.services.orchestration_decision_application as application

    if runtime == "batch":
        seeded = await seed_definition_proposal(sessions, tmp_path)
        goal, user = await _loaded_goal_user(sessions, seeded)
        async with sessions() as db:
            run = await db.get(OrchestrationRun, seeded.run_id)
            decision = await db.get(OrchestrationAuthorityDecision, seeded.decision_id)
    else:
        _, user, goal, run = await _seed(sessions, tmp_path)
    if runtime is True:
        async with sessions() as db:
            decision = await OrchestrationAuthorityDecisionService().create_runtime_question(db, goal.id,
                run_id=run.id, subject="release", authority="human", contract_version="plan:1",
                continuation={"action_type": "continue", "reason": "Approved"},
                question="Proceed?", options=[{"key": "approve"}, {"key": "reject"}])
            await db.commit()
    elif runtime is False:
        decision = await _decision(sessions, goal, run)
    payload = {"selected_option": "approve", **({"contract_version": "plan:1"} if runtime is True else {})}
    if runtime == "batch":
        payload = {"answers": [{"decision_id": str(decision.id), "selected_option": "reject", "reason": "Keep current definition"}]}
    turn, action = await _prepare(sessions, user, goal,
        "answer_agent_definition_batch" if runtime == "batch" else "answer_runtime_question" if runtime else "answer_authority",
        target_id=None if runtime == "batch" else decision.id,
        content=f"Reject {decision.id} because Keep current definition" if runtime == "batch" else f"Answer {decision.id} approve", payload=payload)
    observed = []
    original_lock = OrchestrationService._lock_goal_for_baseline_transition
    depth = 0

    @asynccontextmanager
    async def observed_lock(service, db, goal_id):
        nonlocal depth
        async with original_lock(service, db, goal_id):
            depth += 1
            try:
                yield
            finally:
                if not failure and depth == 1:
                    async with sessions() as fresh:
                        saved = await fresh.get(OrchestrationAuthorityDecision, decision.id)
                        row = await fresh.get(ProjectChatTurn, turn.id)
                        assert saved.status == "answered"
                        assert row.status == "completed" and row.action_receipt["status"] == "recorded"
                        continuations = (await fresh.scalars(select(OrchestrationAction).where(
                            OrchestrationAction.run_id == run.id,
                            OrchestrationAction.action_type == "decision_continuation"))).all()
                        assert len(continuations) == int(runtime is True)
                        observed.append(True)
                depth -= 1

    if runtime is True:
        monkeypatch.setattr(OrchestrationService, "_lock_goal_for_baseline_transition", observed_lock)
    else:
        def forbidden_mutex(*args, **kwargs):
            pytest.fail("ordinary answer acceptance awaited a long transition mutex")
        monkeypatch.setattr(OrchestrationService, "_lock_goal_for_baseline_transition", forbidden_mutex)
    if failure:
        application_name = "answer_agent_review_batch" if runtime == "batch" else "answer_decision"
        original_answer = getattr(application, application_name)

        async def failing_answer(*args, **kwargs):
            async def fail_before_receipt(db, _result):
                assert (await db.get(OrchestrationAuthorityDecision, decision.id)).status == "answered"
                raise RuntimeError("crash after answer before receipt publication")
            kwargs["completion_callback"] = fail_before_receipt
            return await original_answer(*args, **kwargs)
        monkeypatch.setattr(application, application_name, failing_answer)
    async with sessions() as db:
        if runtime is not True and not failure:
            original_commit = db.commit
            async def observed_writer_commit():
                staged = await db.get(OrchestrationAuthorityDecision, decision.id)
                if staged.status == "answered" and not observed:
                    # A fresh reader sees neither half until this actual outer
                    # writer transaction commits the answer+obligation+receipt.
                    async with sessions() as fresh:
                        assert (await fresh.get(OrchestrationAuthorityDecision, decision.id)).status == "pending"
                        assert (await fresh.get(ProjectChatTurn, turn.id)).action_receipt is None
                    await original_commit()
                    async with sessions() as fresh:
                        saved = await fresh.get(OrchestrationAuthorityDecision, decision.id)
                        row = await fresh.get(ProjectChatTurn, turn.id)
                        assert saved.status == "answered"
                        assert row.status == "completed" and row.action_receipt["status"] == "recorded"
                        obligation = await fresh.get(OrchestrationAction, saved.continuation_action_id)
                        assert obligation.action_type == "apply_authority_answer" and obligation.status == "reserved"
                    observed.append(True)
                else:
                    await original_commit()
            monkeypatch.setattr(db, "commit", observed_writer_commit)
        if failure:
            with pytest.raises(RuntimeError, match="crash after answer"):
                async with db.begin():
                    await _service(sessions).execute(db, await db.get(ProjectChatTurn, turn.id), action)
        else:
            await _service(sessions).execute(db, await db.get(ProjectChatTurn, turn.id), action)
    if failure:
        async with sessions() as fresh:
            assert (await fresh.get(OrchestrationAuthorityDecision, decision.id)).status == "pending"
            assert (await fresh.get(ProjectChatTurn, turn.id)).action_receipt is None
            assert (await fresh.scalars(select(OrchestrationAction).where(
                OrchestrationAction.run_id == run.id,
                OrchestrationAction.action_type.in_(["decision_continuation", "apply_authority_answer"])))).all() == []
    else:
        assert observed


@pytest.mark.parametrize("selected_option", ["approve", "request_changes"])
async def test_spr231_chat_request_dies_after_answer_obligation_and_receipt_commit(sessions, tmp_path, monkeypatch, selected_option):
    """SPR230/SPR231 no request coroutine survives; fresh consumer uses the saved obligation."""
    from tests.test_orchestration_decision_application import _spr231_committed_ordinary
    from huddleroom.services import orchestration_decision_application as application
    from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
    project, user, goal, run, source, decision = await _spr231_committed_ordinary(sessions, tmp_path)
    turn, action = await _prepare(sessions, user, goal, "answer_authority", target_id=decision.id,
        content=f"Answer {decision.id} {selected_option}", payload={"selected_option": selected_option})
    class RequestTerminated(BaseException):
        pass
    interrupted = []
    async with sessions() as submitting:
        original_commit = submitting.commit
        async def commit_then_terminate():
            await original_commit()
            async with sessions() as fresh:
                saved = await fresh.get(OrchestrationAuthorityDecision, decision.id)
                chat = await fresh.get(ProjectChatTurn, turn.id)
                if saved.status == "answered" and chat.action_receipt:
                    interrupted.append(True)
                    raise RequestTerminated()
        monkeypatch.setattr(submitting, "commit", commit_then_terminate)
        with pytest.raises(RequestTerminated):
            await _service(sessions).execute(submitting, await submitting.get(ProjectChatTurn, turn.id), action)
    assert interrupted == [True]
    async with sessions() as fresh:
        saved = await fresh.get(OrchestrationAuthorityDecision, decision.id)
        chat = await fresh.get(ProjectChatTurn, turn.id)
        assert saved.status == "answered" and saved.selected_option == selected_option
        assert chat.status == "completed" and chat.action_receipt["status"] == "recorded"
        obligation_id = saved.continuation_action_id
        assert (await fresh.get(OrchestrationAction, obligation_id)).status == "reserved"
    async with sessions() as consumer:
        await application.drain_answer_applications(consumer, run.id)
        await consumer.commit()
    async with sessions() as consumer:
        await application.drain_answer_applications(consumer, run.id)
        await consumer.commit()
    async with sessions() as fresh:
        assert (await fresh.get(OrchestrationAction, obligation_id)).status == "completed"
        assert await fresh.scalar(select(func.count()).select_from(EventLog).where(
            EventLog.dedup_key == f"authority.decision_resolved:{decision.id}")) == 1
