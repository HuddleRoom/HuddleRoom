import pytest
import uuid

from huddleroom.services.orchestration_steering import (
    OrchestrationSteeringService, SteeringDomainError, SteeringDraft, active_direction_ids, normalize_draft,
    steering_versions_from_snapshot,
)
from huddleroom.config import settings
from huddleroom.models.orchestration_steering import OrchestrationSteeringTransition
from huddleroom.models.event_log import EventLog
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationDecision, OrchestrationGoal
from huddleroom.models.orchestration import OrchestrationGate, OrchestrationRoadmapItem, OrchestrationRoadmapVersion
from huddleroom.models.artifact import Artifact
from huddleroom.models.task import Task
from huddleroom.models.orchestration_steering import OrchestrationSteeringResultLink
from huddleroom.models.orchestration_conversation import ConversationMessage, ConversationResponse, conversation_message_id, conversation_response_id, conversation_provider_request_id
from sqlalchemy import func, select


def test_normalize_draft_accepts_exact_stripped_directive_limit():
    draft = SteeringDraft(" " + "x" * 4000 + " ", "goal", " goal ", "run", "remaining_current_run", " impact ")
    normalized = normalize_draft(draft)
    assert normalized.directive == "x" * 4000
    assert normalized.target_id == "goal"


@pytest.mark.parametrize("directive", ["", " " * 2, "x" * 4001])
def test_normalize_draft_rejects_invalid_directive_lengths(directive):
    with pytest.raises(SteeringDomainError, match="steering_invalid_directive"):
        normalize_draft(SteeringDraft(directive, "goal", "goal", "run", "remaining_current_run", "impact"))


def test_snapshot_parsers_reject_malformed_or_boolean_versions():
    assert steering_versions_from_snapshot({"steering": {"versions": {
        "inbox_version": True, "direction_version": 0, "contract_version": "c", "plan_version": "p",
    }}}) is None
    assert active_direction_ids({"steering": {"active_directions": [{"request_id": "not-a-uuid"}]}}) == ()


@pytest.mark.asyncio
async def test_submit_replays_canonical_draft_and_appends_one_transition(db_session, conversation_goal_run, test_project, test_user, monkeypatch):
    monkeypatch.setattr(settings, "orchestration_conversation_steering_enabled", True)
    goal, run = conversation_goal_run
    goal.status, goal.manager_user_id, run.status, run.phase = "active", test_user.id, "running", "authorized"
    service = OrchestrationSteeringService()
    request_id = uuid.uuid4()
    first = await service.submit(db_session, test_project.id, goal.id, test_user.id, request_id, SteeringDraft(" keep scope ", "goal", f" {goal.id} ", "run", "remaining_current_run", " impact "))
    second = await service.submit(db_session, test_project.id, goal.id, test_user.id, request_id, SteeringDraft("keep scope", "goal", str(goal.id), "run", "remaining_current_run", "impact"))
    transitions = (await db_session.scalars(select(OrchestrationSteeringTransition).where(OrchestrationSteeringTransition.request_id == first.id))).all()
    assert second.id == first.id
    assert len(transitions) == 1
    assert (await service.ledger(db_session, test_project.id, goal.id, test_user.id)).inbox_version == 1


@pytest.mark.asyncio
async def test_submit_rejects_unauthorized_before_lifecycle_lookup(db_session, conversation_goal_run, test_project, test_user, monkeypatch):
    monkeypatch.setattr(settings, "orchestration_conversation_steering_enabled", True)
    goal, _ = conversation_goal_run
    with pytest.raises(SteeringDomainError, match="steering_forbidden"):
        await OrchestrationSteeringService().submit(db_session, test_project.id, goal.id, uuid.uuid4(), uuid.uuid4(), SteeringDraft("x", "goal", str(goal.id), "run", "remaining_current_run", "y"))


async def _active_service(db_session, conversation_goal_run, test_user, monkeypatch):
    monkeypatch.setattr(settings, "orchestration_conversation_steering_enabled", True)
    goal, run = conversation_goal_run
    goal.status, goal.manager_user_id, run.status, run.phase = "active", test_user.id, "running", "authorized"
    return OrchestrationSteeringService(), goal, run


@pytest.mark.asyncio
async def test_withdraw_and_process_append_ordered_transitions_and_versions(db_session, conversation_goal_run, test_project, test_user, monkeypatch):
    service, goal, run = await _active_service(db_session, conversation_goal_run, test_user, monkeypatch)
    withdrawn = await service.submit(db_session, test_project.id, goal.id, test_user.id, uuid.uuid4(), SteeringDraft("withdraw", "goal", str(goal.id), "run", "remaining_current_run", "impact"))
    await service.withdraw(db_session, test_project.id, goal.id, test_user.id, withdrawn.id)
    request = await service.submit(db_session, test_project.id, goal.id, test_user.id, uuid.uuid4(), SteeringDraft("apply", "goal", str(goal.id), "run", "remaining_current_run", "impact"))
    result = await service.process_pending(db_session, goal, run)
    ledger = await service.ledger(db_session, test_project.id, goal.id, test_user.id)
    states = (await db_session.scalars(select(OrchestrationSteeringTransition.to_status).where(OrchestrationSteeringTransition.request_id == request.id).order_by(OrchestrationSteeringTransition.sequence))).all()
    assert result.applied_request_ids == (request.id,)
    assert states == ["pending", "being_considered", "applied"]
    assert ledger.inbox_version == 3
    assert ledger.direction_version == 1


@pytest.mark.asyncio
async def test_paused_processing_keeps_pending(db_session, conversation_goal_run, test_project, test_user, monkeypatch):
    service, goal, run = await _active_service(db_session, conversation_goal_run, test_user, monkeypatch)
    request = await service.submit(db_session, test_project.id, goal.id, test_user.id, uuid.uuid4(), SteeringDraft("wait", "goal", str(goal.id), "run", "remaining_current_run", "impact"))
    goal.status, run.status = "paused", "paused"
    assert (await service.process_pending(db_session, goal, run)).processed_request_ids == ()
    assert request.status == "pending"


@pytest.mark.asyncio
async def test_explicit_supersession_orders_old_before_new_and_versions(db_session, conversation_goal_run, test_project, test_user, monkeypatch):
    service, goal, run = await _active_service(db_session, conversation_goal_run, test_user, monkeypatch)
    old = await service.submit(db_session, test_project.id, goal.id, test_user.id, uuid.uuid4(), SteeringDraft("old", "goal", str(goal.id), "run", "remaining_current_run", "impact"))
    await service.process_pending(db_session, goal, run)
    new = await service.submit(db_session, test_project.id, goal.id, test_user.id, uuid.uuid4(), SteeringDraft("new", "goal", str(goal.id), "run", "remaining_current_run", "impact", supersedes_request_id=old.id))
    await service.process_pending(db_session, goal, run)
    old_states = (await db_session.scalars(select(OrchestrationSteeringTransition.to_status).where(OrchestrationSteeringTransition.request_id == old.id).order_by(OrchestrationSteeringTransition.sequence))).all()
    new_states = (await db_session.scalars(select(OrchestrationSteeringTransition.to_status).where(OrchestrationSteeringTransition.request_id == new.id).order_by(OrchestrationSteeringTransition.sequence))).all()
    ledger = await service.ledger(db_session, test_project.id, goal.id, test_user.id)
    assert old_states == ["pending", "being_considered", "applied", "superseded"]
    assert new_states == ["pending", "being_considered", "applied"]
    assert (ledger.inbox_version, ledger.direction_version) == (3, 3)


@pytest.mark.asyncio
async def test_context_versions_and_goal_lifetime_survive_current_run(db_session, conversation_goal_run, test_project, test_user, monkeypatch):
    service, goal, run = await _active_service(db_session, conversation_goal_run, test_user, monkeypatch)
    request = await service.submit(db_session, test_project.id, goal.id, test_user.id, uuid.uuid4(), SteeringDraft("future", "goal", str(goal.id), "goal", "future_runs", "impact"))
    await service.process_pending(db_session, goal, run)
    snapshot = await service.context_snapshot(db_session, goal, run)
    versions = await service.current_versions(db_session, goal, run)
    assert snapshot["active_directions"][0]["request_id"] == str(request.id)
    await service.assert_current_versions(db_session, goal, run, versions)


@pytest.mark.asyncio
async def test_proposal_replays_promotes_and_disabled_dismiss_is_guarded(db_session, conversation_goal_run, test_project, test_user, monkeypatch):
    service, goal, run = await _active_service(db_session, conversation_goal_run, test_user, monkeypatch)
    message_id = conversation_message_id(goal.id, test_user.id, uuid.uuid4())
    message = ConversationMessage(id=message_id, goal_id=goal.id, actor_id=test_user.id, client_request_id=uuid.uuid4(), sequence=1, content="q")
    response = ConversationResponse(id=conversation_response_id(message_id), message_id=message_id, run_id=run.id, dossier={}, context_manifest={}, context_version="v", provider_request_id=conversation_provider_request_id(conversation_response_id(message_id)))
    db_session.add(message); await db_session.flush(); db_session.add(response); await db_session.flush()
    draft = SteeringDraft("proposal", "goal", str(goal.id), "run", "remaining_current_run", "impact")
    proposal = await service.persist_proposal(db_session, message, response, draft)
    assert (await service.persist_proposal(db_session, message, response, draft)).id == proposal.id
    proposal_ledger = await service.ledger(db_session, test_project.id, goal.id, test_user.id)
    assert proposal_ledger.requests == ()
    assert proposal_ledger.inbox_version == proposal_ledger.direction_version == 0
    request = await service.submit(db_session, test_project.id, goal.id, test_user.id, uuid.uuid4(), SteeringDraft("proposal", "goal", str(goal.id), "run", "remaining_current_run", "impact", source_proposal_id=proposal.id))
    assert proposal.promoted_request_id == request.id
    submitted_ledger = await service.ledger(db_session, test_project.id, goal.id, test_user.id)
    assert submitted_ledger.requests == (request,)
    assert submitted_ledger.inbox_version == 1
    monkeypatch.setattr(settings, "orchestration_conversation_steering_enabled", False)
    with pytest.raises(SteeringDomainError, match="steering_disabled"):
        await service.dismiss_proposal(db_session, test_project.id, goal.id, test_user.id, proposal.id)


@pytest.mark.asyncio
async def test_submit_events_are_exactly_once_on_replay(db_session, conversation_goal_run, test_project, test_user, monkeypatch):
    service, goal, _ = await _active_service(db_session, conversation_goal_run, test_user, monkeypatch)
    client = uuid.uuid4(); draft = SteeringDraft("event", "goal", str(goal.id), "run", "remaining_current_run", "impact")
    await service.submit(db_session, test_project.id, goal.id, test_user.id, client, draft)
    await service.submit(db_session, test_project.id, goal.id, test_user.id, client, draft)
    count = await db_session.scalar(select(func.count(EventLog.id)).where(EventLog.project_id == test_project.id, EventLog.event_type == "orchestration.steering_changed"))
    assert count == 1


@pytest.mark.asyncio
async def test_result_link_replays_valid_lineage_and_rejects_mismatch(db_session, conversation_goal_run, test_project, test_user, monkeypatch):
    service, goal, run = await _active_service(db_session, conversation_goal_run, test_user, monkeypatch)
    request = await service.submit(db_session, test_project.id, goal.id, test_user.id, uuid.uuid4(), SteeringDraft("link", "goal", str(goal.id), "run", "remaining_current_run", "impact"))
    decision = OrchestrationDecision(run_id=run.id, decision_type="m7", input_snapshot={}, parsed_decision={})
    db_session.add(decision); await db_session.flush()
    action = OrchestrationAction(run_id=run.id, decision_id=decision.id, idempotency_key="link-test", action_type="noop", request={})
    db_session.add(action); await db_session.flush()
    link = await service.link_result(db_session, request.id, decision.id, action.id)
    assert (await service.link_result(db_session, request.id, decision.id, action.id)).id == link.id
    bad_action = OrchestrationAction(run_id=run.id, decision_id=None, idempotency_key="bad-link", action_type="noop", request={})
    db_session.add(bad_action); await db_session.flush()
    with pytest.raises(SteeringDomainError):
        await service.link_result(db_session, request.id, decision.id, bad_action.id)
    assert await db_session.scalar(select(func.count(OrchestrationSteeringResultLink.id))) == 1


@pytest.mark.asyncio
async def test_invalid_proposal_promotion_leaves_no_request_or_version(db_session, conversation_goal_run, test_project, test_user, monkeypatch):
    service, goal, _ = await _active_service(db_session, conversation_goal_run, test_user, monkeypatch)
    with pytest.raises(SteeringDomainError, match="proposal_not_promotable"):
        await service.submit(db_session, test_project.id, goal.id, test_user.id, uuid.uuid4(), SteeringDraft("bad", "goal", str(goal.id), "run", "remaining_current_run", "impact", source_proposal_id=uuid.uuid4()))
    assert (await service.ledger(db_session, test_project.id, goal.id, test_user.id)).inbox_version == 0


@pytest.mark.asyncio
async def test_expanded_plan_item_defers_after_task_starts(db_session, conversation_goal_run, test_project, test_user, monkeypatch):
    service, goal, run = await _active_service(db_session, conversation_goal_run, test_user, monkeypatch)
    task = Task(project_id=test_project.id, title="expanded", status="ready", metadata_={"orchestration": {"goal_id": str(goal.id), "run_id": str(run.id)}})
    db_session.add(task); await db_session.flush()
    run.plan_state = {"accepted_plan_snapshot": {"items": [{"id": "item-1"}]}, "expanded_items": [{"plan_item_id": "item-1", "task_id": str(task.id)}]}
    request = await service.submit(db_session, test_project.id, goal.id, test_user.id, uuid.uuid4(), SteeringDraft("item", "plan_item", "item-1", "item", "selected_item", "impact"))
    task.status = "in_progress"
    await service.process_pending(db_session, goal, run)
    assert request.status == "deferred"


async def _roadmap_version(db_session, project_id, goal, run, version, items):
    artifact = Artifact(project_id=project_id, name=f"Roadmap {version}", artifact_type="plan", metadata_={})
    db_session.add(artifact); await db_session.flush()
    roadmap_version = OrchestrationRoadmapVersion(
        goal_id=goal.id, run_id=run.id, version=version, plan_artifact_id=artifact.id,
        snapshot={"items": items}, fingerprint=f"{version:064x}", approval_reference={},
    )
    db_session.add(roadmap_version); await db_session.flush()
    return roadmap_version


@pytest.mark.asyncio
async def test_roadmap_snapshot_allows_unreleased_and_retained_unstarted_items(db_session, conversation_goal_run, test_project, test_user, monkeypatch):
    service, goal, run = await _active_service(db_session, conversation_goal_run, test_user, monkeypatch)
    goal.goal_type = "roadmap"
    first = await _roadmap_version(db_session, test_project.id, goal, run, 1, [{"item_key": "retained"}])
    task = Task(project_id=test_project.id, title="retained", status="ready", metadata_={"orchestration": {"goal_id": str(goal.id), "run_id": str(run.id)}})
    gate = OrchestrationGate(run_id=run.id, success_criterion_key="retained", gate_type="item", required_evidence={})
    db_session.add_all((task, gate)); await db_session.flush()
    db_session.add(OrchestrationRoadmapItem(goal_id=goal.id, first_version_id=first.id, item_key="retained", unit_type="task", item_snapshot={}, task_id=task.id, gate_id=gate.id)); await db_session.flush()
    await _roadmap_version(db_session, test_project.id, goal, run, 2, [{"item_key": "retained"}, {"item_key": "unreleased"}])
    retained = await service.submit(db_session, test_project.id, goal.id, test_user.id, uuid.uuid4(), SteeringDraft("retained", "plan_item", "retained", "item", "selected_item", "impact"))
    unreleased = await service.submit(db_session, test_project.id, goal.id, test_user.id, uuid.uuid4(), SteeringDraft("unreleased", "plan_item", "unreleased", "item", "selected_item", "impact"))
    assert {retained.target_id, unreleased.target_id} == {"retained", "unreleased"}


@pytest.mark.asyncio
async def test_roadmap_released_or_child_item_expires_and_item_ownership_drift_defers(db_session, conversation_goal_run, test_project, test_user, monkeypatch):
    service, goal, run = await _active_service(db_session, conversation_goal_run, test_user, monkeypatch)
    goal.goal_type = "roadmap"
    version = await _roadmap_version(db_session, test_project.id, goal, run, 1, [{"item_key": "released"}, {"item_key": "child"}])
    task = Task(project_id=test_project.id, title="released", status="ready", metadata_={"orchestration": {"goal_id": str(goal.id), "run_id": str(run.id)}})
    child = OrchestrationGoal(project_id=test_project.id, objective="child", original_request="child", success_criteria=[], constraints={}, budget={})
    gate = OrchestrationGate(run_id=run.id, success_criterion_key="released", gate_type="item", required_evidence={})
    child_gate = OrchestrationGate(run_id=run.id, success_criterion_key="child", gate_type="item", required_evidence={})
    db_session.add_all((task, child, gate, child_gate)); await db_session.flush()
    db_session.add_all((
        OrchestrationRoadmapItem(goal_id=goal.id, first_version_id=version.id, item_key="released", unit_type="task", item_snapshot={}, task_id=task.id, gate_id=gate.id),
        OrchestrationRoadmapItem(goal_id=goal.id, first_version_id=version.id, item_key="child", unit_type="goal", item_snapshot={}, child_goal_id=child.id, gate_id=child_gate.id),
    )); await db_session.flush()
    request = await service.submit(db_session, test_project.id, goal.id, test_user.id, uuid.uuid4(), SteeringDraft("released", "plan_item", "released", "item", "selected_item", "impact"))
    task.metadata_ = {"orchestration": {"goal_id": str(goal.id), "run_id": str(uuid.uuid4())}}
    await service.process_pending(db_session, goal, run)
    assert request.status == "deferred"
    with pytest.raises(SteeringDomainError, match="steering_invalid_target"):
        await service.submit(db_session, test_project.id, goal.id, test_user.id, uuid.uuid4(), SteeringDraft("child", "plan_item", "child", "item", "selected_item", "impact"))


@pytest.mark.asyncio
async def test_submit_rejects_started_or_wrong_owner_task_and_expanded_item(db_session, conversation_goal_run, test_project, test_user, monkeypatch):
    service, goal, run = await _active_service(db_session, conversation_goal_run, test_user, monkeypatch)
    started = Task(project_id=test_project.id, title="started", status="in_progress", metadata_={"orchestration": {"goal_id": str(goal.id), "run_id": str(run.id)}})
    drifted = Task(project_id=test_project.id, title="drifted", status="ready", metadata_={"orchestration": {"goal_id": str(goal.id), "run_id": str(uuid.uuid4())}})
    db_session.add_all((started, drifted)); await db_session.flush()
    run.plan_state = {"accepted_plan_snapshot": {"items": [{"id": "expanded"}]}, "expanded_items": [{"plan_item_id": "expanded", "task_id": str(started.id)}]}
    for draft in (
        SteeringDraft("started task", "task", str(started.id), "item", "selected_item", "impact"),
        SteeringDraft("drifted task", "task", str(drifted.id), "item", "selected_item", "impact"),
        SteeringDraft("started expanded", "plan_item", "expanded", "item", "selected_item", "impact"),
    ):
        with pytest.raises(SteeringDomainError, match="steering_invalid_target"):
            await service.submit(db_session, test_project.id, goal.id, test_user.id, uuid.uuid4(), draft)
