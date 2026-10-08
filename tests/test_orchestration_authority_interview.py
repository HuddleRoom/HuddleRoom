import uuid
import json

import pytest
from fastapi import HTTPException
from sqlalchemy import select

pytestmark = pytest.mark.usefixtures("safe_goal_analysis", "safe_agent_definition_review")

from huddleroom.models.agent import Agent
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGoal, OrchestrationRun
from huddleroom.models.task import Task
from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService


def _agent(name_prefix: str = "manager", *, is_active: bool = True) -> Agent:
    return Agent(
        name=f"{name_prefix}-{uuid.uuid4()}",
        role="manager",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=["coordination"],
        config={},
        is_active=is_active,
    )


@pytest.fixture
def make_goal(test_project):
    def factory(*, weight: str = "standard") -> OrchestrationGoal:
        return OrchestrationGoal(
            id=uuid.uuid4(),
            project_id=test_project.id,
            objective="Ship the authority interview feature",
            success_criteria=[{"key": "done", "description": "Done"}],
            weight=weight,
        )

    return factory


@pytest.fixture
def make_run():
    def factory(goal: OrchestrationGoal) -> OrchestrationRun:
        return OrchestrationRun(id=uuid.uuid4(), goal_id=goal.id, status="running")

    return factory


@pytest.mark.asyncio
async def test_link_delegation_action_sets_related_action_id_once(db_session, make_goal, make_run):
    goal = make_goal()
    run = make_run(goal)
    db_session.add_all([goal, run])
    await db_session.flush()

    decision_service = OrchestrationAuthorityDecisionService()
    manager = _agent()
    db_session.add(manager)
    await db_session.flush()
    decision = await decision_service.create_pending(
        db_session,
        goal.id,
        decision_key="test:link",
        title="Approve?",
        question="Approve the plan?",
        authority="manager",
        authority_agent_id=manager.id,
        options=["approve", "reject"],
        run_id=run.id,
    )
    action = OrchestrationAction(
        run_id=run.id,
        idempotency_key="test:link-action",
        action_type="create_delegation_task",
    )
    db_session.add(action)
    await db_session.flush()
    action_id = action.id

    linked = await decision_service.link_delegation_action(db_session, decision, action_id=action_id)
    assert linked.related_action_id == action_id

    # Idempotent no-op: linking the same action_id again doesn't error or change anything.
    relinked = await decision_service.link_delegation_action(db_session, decision, action_id=action_id)
    assert relinked.related_action_id == action_id


from huddleroom.schemas.orchestration import OrchestrationGoalCreate
from huddleroom.services.orchestration_service import OrchestrationService


@pytest.mark.asyncio
async def test_cancel_goal_cancels_its_pending_decisions(db_session, test_project):
    service = OrchestrationService()
    goal, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Authority interview cancellation test",
            success_criteria=[{"key": "done", "description": "Done"}],
        ),
        created_by_user_id=None,
    )
    decision_service = OrchestrationAuthorityDecisionService()
    decision = await decision_service.create_pending(
        db_session,
        goal.id,
        decision_key="test:cancel-cascade",
        title="Clarify scope",
        question="What does done mean?",
        authority="human",
        run_id=run.id,
    )

    await service.cancel_goal(db_session, test_project.id, goal.id)

    await db_session.refresh(decision)
    assert decision.status == "cancelled"
    assert decision.reason == "goal cancelled"


from datetime import datetime, timedelta, timezone

from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision, OrchestrationWarning
from huddleroom.services.orchestration_authority_interview import build_checkpoint


def _decision(*, authority="human", asked_at=None) -> OrchestrationAuthorityDecision:
    return OrchestrationAuthorityDecision(
        id=uuid.uuid4(),
        goal_id=uuid.uuid4(),
        decision_key=str(uuid.uuid4()),
        title="t",
        authority=authority,
        question="q",
        options=[],
        asked_at=asked_at or datetime.now(timezone.utc),
        created_at=asked_at or datetime.now(timezone.utc),
    )


def test_build_checkpoint_caps_at_max_questions_oldest_first():
    base = datetime.now(timezone.utc)
    decisions = [_decision(asked_at=base + timedelta(seconds=i)) for i in range(7)]

    checkpoint, deferred = build_checkpoint(decisions, max_questions=5)

    assert [d.id for d in checkpoint] == [d.id for d in decisions[:5]]
    assert [d.id for d in deferred] == [d.id for d in decisions[5:]]


def test_build_checkpoint_excludes_non_human_authority():
    human = _decision(authority="human")
    manager = _decision(authority="manager")

    checkpoint, deferred = build_checkpoint([human, manager], max_questions=5)

    assert checkpoint == [human]
    assert deferred == []


def test_build_checkpoint_rejects_non_positive_max_questions():
    with pytest.raises(ValueError, match="max_questions must be >= 1"):
        build_checkpoint([], max_questions=0)


def test_build_checkpoint_never_defers_agent_definition_review_proposals():
    """Given N agent_definition_review:proposal decisions with N > max_questions:
    all are in checkpoint, none deferred."""
    base = datetime.now(timezone.utc)
    # Create 9 proposal decisions
    proposals = [
        OrchestrationAuthorityDecision(
            id=uuid.uuid4(),
            goal_id=uuid.uuid4(),
            decision_key=f"agent_definition_review:proposal:{uuid.uuid4()}",
            title=f"Agent {i}",
            authority="human",
            question=f"Approve agent {i}?",
            options=[],
            asked_at=base + timedelta(seconds=i),
            created_at=base + timedelta(seconds=i),
        )
        for i in range(9)
    ]

    checkpoint, deferred = build_checkpoint(proposals, max_questions=5)

    # All 9 proposals must be in checkpoint
    assert len(checkpoint) == 9
    assert all(d.decision_key.startswith("agent_definition_review:proposal:") for d in checkpoint)
    assert deferred == []


def test_build_checkpoint_caps_non_proposals_but_keeps_all_proposals():
    """Mixed: some proposal decisions + some non-proposal human decisions,
    with non-proposal count > max_questions: ALL proposal decisions are in
    checkpoint, and the non-proposal ones are capped at max_questions."""
    base = datetime.now(timezone.utc)
    # 3 proposals
    proposals = [
        OrchestrationAuthorityDecision(
            id=uuid.uuid4(),
            goal_id=uuid.uuid4(),
            decision_key=f"agent_definition_review:proposal:{uuid.uuid4()}",
            title=f"Agent {i}",
            authority="human",
            question=f"Approve agent {i}?",
            options=[],
            asked_at=base + timedelta(seconds=i),
            created_at=base + timedelta(seconds=i),
        )
        for i in range(3)
    ]
    # 8 regular questions (asked after proposals)
    regular = [
        OrchestrationAuthorityDecision(
            id=uuid.uuid4(),
            goal_id=uuid.uuid4(),
            decision_key=f"clarification:{i}",
            title=f"Question {i}",
            authority="human",
            question=f"What about {i}?",
            options=[],
            asked_at=base + timedelta(seconds=10 + i),
            created_at=base + timedelta(seconds=10 + i),
        )
        for i in range(8)
    ]
    all_decisions = proposals + regular

    checkpoint, deferred = build_checkpoint(all_decisions, max_questions=5)

    # All 3 proposals + first 5 regular questions = 8 in checkpoint
    assert len(checkpoint) == 8
    proposals_in_checkpoint = [d for d in checkpoint if d.decision_key.startswith("agent_definition_review:proposal:")]
    regulars_in_checkpoint = [d for d in checkpoint if not d.decision_key.startswith("agent_definition_review:proposal:")]
    assert len(proposals_in_checkpoint) == 3
    assert len(regulars_in_checkpoint) == 5
    # Remaining 3 regular questions are deferred
    assert len(deferred) == 3
    assert all(not d.decision_key.startswith("agent_definition_review:proposal:") for d in deferred)


def test_build_checkpoint_preserves_pure_non_proposal_capping():
    """Preserve existing behavior for pure non-proposal case: cap still applies."""
    base = datetime.now(timezone.utc)
    decisions = [_decision(asked_at=base + timedelta(seconds=i)) for i in range(7)]

    checkpoint, deferred = build_checkpoint(decisions, max_questions=5)

    assert [d.id for d in checkpoint] == [d.id for d in decisions[:5]]
    assert [d.id for d in deferred] == [d.id for d in decisions[5:]]


@pytest.mark.asyncio
async def test_build_checkpoint_after_rerun_excludes_superseded_process_questions(db_session, make_goal, make_run):
    """Closes the Check-list item 'rerunning one process produces a focused
    checkpoint (no re-asking completed processes)'. The invariant this
    relies on isn't in build_checkpoint itself -- it's that a rerun cancels
    the old process's pending decision (e.g. TeamHierarchyProcess's
    `_cancel_pending_for_run`) before raising a new one, so build_checkpoint,
    which only ever sees `status="pending"` rows, naturally never surfaces
    the superseded question. This asserts that end-to-end instead of taking
    it on faith."""
    goal = make_goal()
    run = make_run(goal)
    db_session.add_all([goal, run])
    await db_session.flush()
    decision_service = OrchestrationAuthorityDecisionService()

    stale = await decision_service.create_pending(
        db_session, goal.id, decision_key="rerun:approval", title="Approve v1",
        question="Approve the first proposal?", authority="human", run_id=run.id,
    )
    await decision_service.cancel_decision(db_session, stale, reason="process rerun superseded this decision")
    fresh = await decision_service.create_pending(
        db_session, goal.id, decision_key="rerun:approval", title="Approve v2",
        question="Approve the revised proposal?", authority="human", run_id=run.id,
    )

    pending = await decision_service.list_decisions(db_session, goal.id, status="pending")
    checkpoint, deferred = build_checkpoint(pending, max_questions=5)

    assert [d.id for d in checkpoint] == [fresh.id]
    assert deferred == []


from huddleroom.services.orchestration_authority_interview import sync_deferred_questions_memory
from huddleroom.services.orchestration_memory_service import OrchestrationMemoryService


@pytest.mark.asyncio
async def test_sync_deferred_questions_memory_writes_open_questions_section(db_session, make_goal, make_run):
    goal = make_goal()
    run = make_run(goal)
    db_session.add_all([goal, run])
    await db_session.flush()
    deferred = [
        OrchestrationAuthorityDecision(
            id=uuid.uuid4(),
            goal_id=goal.id,
            decision_key="deferred-1",
            title="Clarify budget",
            authority="human",
            question="What is the max token budget?",
            options=[],
        )
    ]

    await sync_deferred_questions_memory(db_session, goal, run, deferred)

    section = await OrchestrationMemoryService().get_section(db_session, goal.project_id, goal.id, "open_questions")
    assert section is not None
    assert "Clarify budget" in section.body
    assert section.summary == "1 question(s) deferred from the kickoff checkpoint."


@pytest.mark.asyncio
async def test_sync_deferred_questions_memory_clears_stale_content_when_empty(db_session, make_goal, make_run):
    goal = make_goal()
    run = make_run(goal)
    db_session.add_all([goal, run])
    await db_session.flush()
    stale = OrchestrationAuthorityDecision(
        id=uuid.uuid4(), goal_id=goal.id, decision_key="d1", title="Stale", authority="human",
        question="q", options=[],
    )
    await sync_deferred_questions_memory(db_session, goal, run, [stale])

    await sync_deferred_questions_memory(db_session, goal, run, [])

    section = await OrchestrationMemoryService().get_section(db_session, goal.project_id, goal.id, "open_questions")
    assert "Stale" not in section.body


@pytest.mark.asyncio
async def test_sync_deferred_questions_memory_skips_write_when_empty_and_no_section_exists(
    db_session, make_goal, make_run
):
    goal = make_goal()
    run = make_run(goal)
    db_session.add_all([goal, run])
    await db_session.flush()

    await sync_deferred_questions_memory(db_session, goal, run, [])

    section = await OrchestrationMemoryService().get_section(db_session, goal.project_id, goal.id, "open_questions")
    assert section is None


from huddleroom.schemas.orchestration import (
    OrchestrationAuthorityDecisionResponse,
    OrchestrationCheckpointResponse,
    OrchestrationDecisionAnswerRequest,
    OrchestrationDecisionCancelRequest,
)


def test_decision_answer_request_requires_selected_option():
    with pytest.raises(Exception):
        OrchestrationDecisionAnswerRequest()


@pytest.mark.asyncio
async def test_answer_endpoint_routes_agent_definition_edit_fields_as_legacy_reason(
    client, db_session, test_project, make_goal, make_run
):
    goal = make_goal()
    run = make_run(goal)
    db_session.add_all([goal, run])
    await db_session.flush()
    decision = await OrchestrationAuthorityDecisionService().create_pending(
        db_session, goal.id, decision_key=f"agent_definition_review:proposal:{uuid.uuid4()}",
        title="Edit", question="Edit?", authority="human", options=["edit"], run_id=run.id,
    )

    response = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/decisions/{decision.id}/answer",
        json={"selected_option": "edit", "edited_description": "  Better description.  ", "edited_persona": " Better persona. "},
    )

    assert response.status_code == 200, response.text
    assert json.loads(response.json()["decision"]["reason"]) == {
        "description": "Better description.", "persona": "Better persona.",
    }


@pytest.mark.asyncio
async def test_answer_endpoint_rejects_invalid_agent_definition_edit_without_answering(
    client, db_session, test_project, make_goal, make_run
):
    goal = make_goal()
    run = make_run(goal)
    db_session.add_all([goal, run])
    await db_session.flush()
    decision = await OrchestrationAuthorityDecisionService().create_pending(
        db_session, goal.id, decision_key=f"agent_definition_review:proposal:{uuid.uuid4()}",
        title="Edit", question="Edit?", authority="human", options=["edit"], run_id=run.id,
    )

    response = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/decisions/{decision.id}/answer",
        json={"selected_option": "edit", "edited_description": " ", "edited_persona": "persona"},
    )

    assert response.status_code == 400
    await db_session.refresh(decision)
    assert decision.status == "pending"

    missing_field = await OrchestrationAuthorityDecisionService().create_pending(
        db_session, goal.id, decision_key=f"agent_definition_review:proposal:{uuid.uuid4()}",
        title="Edit", question="Edit?", authority="human", options=["edit"], run_id=run.id,
    )
    response = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/decisions/{missing_field.id}/answer",
        json={"selected_option": "edit", "edited_description": "description"},
    )
    assert response.status_code == 400
    await db_session.refresh(missing_field)
    assert missing_field.status == "pending"

    other_decision = await OrchestrationAuthorityDecisionService().create_pending(
        db_session, goal.id, decision_key="other:decision", title="Other", question="Other?",
        authority="human", options=["approve"], run_id=run.id,
    )
    response = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/decisions/{other_decision.id}/answer",
        json={"selected_option": "approve", "edited_description": "description", "edited_persona": "persona"},
    )
    assert response.status_code == 400
    await db_session.refresh(other_decision)
    assert other_decision.status == "pending"


@pytest.mark.asyncio
async def test_answer_endpoint_rejects_agent_definition_edit_without_literal_edit_option(
    client, db_session, test_project, make_goal, make_run
):
    goal = make_goal()
    run = make_run(goal)
    db_session.add_all([goal, run])
    await db_session.flush()
    decision = await OrchestrationAuthorityDecisionService().create_pending(
        db_session, goal.id, decision_key=f"agent_definition_review:proposal:{uuid.uuid4()}",
        title="Edit", question="Edit?", authority="human", options=[{"key": "edit"}], run_id=run.id,
    )

    response = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/decisions/{decision.id}/answer",
        json={"selected_option": "edit", "edited_description": "description", "edited_persona": "persona"},
    )

    assert response.status_code == 400
    await db_session.refresh(decision)
    assert decision.status == "pending"


@pytest.mark.asyncio
async def test_answer_decision_endpoint_answers_pending_human_decision(client, db_session, test_project):
    service = OrchestrationService()
    goal, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Answer endpoint test",
            success_criteria=[{"key": "done", "description": "Done"}],
        ),
        created_by_user_id=None,
    )
    decision = await OrchestrationAuthorityDecisionService().create_pending(
        db_session, goal.id, decision_key="test:answer-endpoint", title="Clarify",
        question="What is done?", authority="human", options=["ship_now", "ship_later"], run_id=run.id,
    )
    await db_session.flush()

    resp = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/decisions/{decision.id}/answer",
        json={"selected_option": "ship_now", "reason": "Deadline is today."},
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["decision"]["status"] == "answered"
    assert body["decision"]["selected_option"] == "ship_now"
    assert body["process"] is None


@pytest.mark.asyncio
async def test_answer_endpoint_ticks_only_the_answered_baseline_process(
    client, db_session, test_project, monkeypatch
):
    service = OrchestrationService()
    goal, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Answer a baseline clarification",
            success_criteria=[{"key": "done", "description": "Done"}],
        ),
        created_by_user_id=None,
    )
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService
    from huddleroom.services.orchestration_debug_service import OrchestrationDebugService

    async def prohibited_path(*_args, **_kwargs):
        raise AssertionError("answering a decision must not use tick or rerun")

    monkeypatch.setattr(OrchestrationService, "tick", prohibited_path)
    monkeypatch.setattr(OrchestrationDebugService, "rerun_last", prohibited_path)

    process_service = OrchestrationProcessService()
    source_process = await process_service.start_process(
        db_session,
        goal.id,
        process_type="goal_definition",
        trigger_reason="test setup",
        run_id=run.id,
    )
    await process_service.park_process(db_session, source_process)
    decision = await OrchestrationAuthorityDecisionService().create_pending(
        db_session,
        goal.id,
        decision_key="goal_definition:adaptive:1:0:objective",
        title="Clarify goal definition",
        question="What should the goal accomplish?",
        authority="human",
        options=["ship_now"],
        run_id=run.id,
        source_process_run_id=source_process.id,
    )
    await db_session.flush()

    response = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/decisions/{decision.id}/answer",
        json={"selected_option": "ship_now", "reason": "Deadline is today."},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["decision"]["status"] == "answered"
    assert body["process"]["status"] in {"completed", "waiting_decision"}
    manager_selection = await process_service.get_current(db_session, goal.id, "manager_selection")
    assert manager_selection is None


@pytest.mark.asyncio
async def test_answer_endpoint_does_not_advance_a_stale_source_process(client, db_session, test_project):
    service = OrchestrationService()
    goal, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Do not advance a replacement process",
            success_criteria=[{"key": "done", "description": "Done"}],
        ),
        created_by_user_id=None,
    )
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    process_service = OrchestrationProcessService()
    source_process = await process_service.start_process(
        db_session,
        goal.id,
        process_type="goal_definition",
        trigger_reason="test setup",
        run_id=run.id,
    )
    await process_service.park_process(db_session, source_process)
    decision = await OrchestrationAuthorityDecisionService().create_pending(
        db_session,
        goal.id,
        decision_key="goal_definition:adaptive:1:0:objective",
        title="Clarify goal definition",
        question="What should the goal accomplish?",
        authority="human",
        options=["ship_now"],
        run_id=run.id,
        source_process_run_id=source_process.id,
    )
    replacement = await process_service.start_process(
        db_session,
        goal.id,
        process_type="goal_definition",
        trigger_reason="replacement",
        run_id=run.id,
        supersede_waiting=True,
    )
    await db_session.flush()

    response = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/decisions/{decision.id}/answer",
        json={"selected_option": "ship_now"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["process"] is None
    await db_session.refresh(replacement)
    assert replacement.status == "running"


@pytest.mark.asyncio
async def test_answer_endpoint_ignores_an_unsupported_source_process(client, db_session, test_project):
    service = OrchestrationService()
    goal, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Ignore unknown source process",
            success_criteria=[{"key": "done", "description": "Done"}],
        ),
        created_by_user_id=None,
    )
    from huddleroom.models.orchestration_process import OrchestrationProcessRun

    source_process = OrchestrationProcessRun(
        goal_id=goal.id,
        run_id=run.id,
        process_type="unsupported_source",
        status="waiting_decision",
        trigger_reason="test setup",
        input_snapshot={},
        outputs={},
    )
    db_session.add(source_process)
    await db_session.flush()
    decision = await OrchestrationAuthorityDecisionService().create_pending(
        db_session,
        goal.id,
        decision_key="test:unsupported-source",
        title="Unsupported source",
        question="Should this be recorded?",
        authority="human",
        options=["yes"],
        run_id=run.id,
        source_process_run_id=source_process.id,
    )
    await db_session.flush()

    response = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/decisions/{decision.id}/answer",
        json={"selected_option": "yes"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["decision"]["status"] == "answered"
    assert response.json()["process"] is None


@pytest.mark.asyncio
async def test_answer_endpoint_preserves_answer_when_source_run_is_not_tickable(client, db_session, test_project):
    service = OrchestrationService()
    goal, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Keep the answer when execution is paused",
            success_criteria=[{"key": "done", "description": "Done"}],
        ),
        created_by_user_id=None,
    )
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    process_service = OrchestrationProcessService()
    source_process = await process_service.start_process(
        db_session,
        goal.id,
        process_type="goal_definition",
        trigger_reason="test setup",
        run_id=run.id,
    )
    await process_service.park_process(db_session, source_process)
    decision = await OrchestrationAuthorityDecisionService().create_pending(
        db_session,
        goal.id,
        decision_key="goal_definition:adaptive:1:0:objective",
        title="Clarify goal definition",
        question="What should the goal accomplish?",
        authority="human",
        options=["ship_now"],
        run_id=run.id,
        source_process_run_id=source_process.id,
    )
    run.status = "paused"
    await db_session.flush()

    response = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/decisions/{decision.id}/answer",
        json={"selected_option": "ship_now"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["decision"]["status"] == "answered"
    assert response.json()["process"] is None
    await db_session.refresh(source_process)
    assert source_process.status == "waiting_decision"


@pytest.mark.asyncio
async def test_answer_decision_endpoint_guards_nulled_authority_agent(client, db_session, test_project):
    service = OrchestrationService()
    goal, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Nulled agent test",
            success_criteria=[{"key": "done", "description": "Done"}],
        ),
        created_by_user_id=None,
    )
    manager = Agent(
        name=f"manager-{uuid.uuid4()}", role="manager", provider="openai", model="gpt-4o-mini",
        adapter_type="api", capabilities=["coordination"], config={}, is_active=True,
    )
    db_session.add(manager)
    await db_session.flush()
    decision = await OrchestrationAuthorityDecisionService().create_pending(
        db_session, goal.id, decision_key="test:nulled-agent", title="Approve",
        question="Approve hierarchy?", authority="manager", authority_agent_id=manager.id,
        options=["approve"], run_id=run.id,
    )
    await db_session.delete(manager)
    await db_session.flush()
    await db_session.refresh(decision)
    assert decision.authority_agent_id is None  # FK SET NULL confirmed
    await db_session.flush()

    resp = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/decisions/{decision.id}/answer",
        json={"selected_option": "approve"},
    )

    assert resp.status_code == 409
    assert "cancel" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_checkpoint_endpoint_caps_and_reports_deferred_count(client, db_session, test_project):
    service = OrchestrationService()
    goal, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Checkpoint cap test",
            success_criteria=[{"key": "done", "description": "Done"}],
        ),
        created_by_user_id=None,
    )
    decision_service = OrchestrationAuthorityDecisionService()
    for i in range(7):
        await decision_service.create_pending(
            db_session, goal.id, decision_key=f"test:checkpoint-{i}", title=f"Q{i}",
            question=f"Question {i}?", authority="human", run_id=run.id,
        )
    await db_session.flush()

    resp = await client.get(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/decisions/checkpoint"
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["items"]) == 5
    assert body["deferred_count"] == 2
    assert body["max_questions"] == 5


from huddleroom.models.artifact import Artifact


@pytest.mark.asyncio
async def test_execute_record_authority_decision_action_answers_from_decision_report(db_session, test_project):
    service = OrchestrationService()
    goal, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Record authority decision test",
            success_criteria=[{"key": "done", "description": "Done"}],
        ),
        created_by_user_id=None,
    )
    manager = Agent(
        name=f"manager-{uuid.uuid4()}", role="manager", provider="openai", model="gpt-4o-mini",
        adapter_type="api", capabilities=["coordination"], config={}, is_active=True,
    )
    db_session.add(manager)
    await db_session.flush()
    decision = await OrchestrationAuthorityDecisionService().create_pending(
        db_session, goal.id, decision_key="test:record-1", title="Approve hierarchy",
        question="Approve?", authority="manager", authority_agent_id=manager.id,
        options=["approve", "reject"], run_id=run.id,
    )
    delegation_action = await service.execute_create_delegation_task_action(
        db_session,
        run_id=run.id,
        request={
            "action_type": "create_delegation_task",
            "agent_id": str(manager.id),
            "work_function": "decision",
            "scope": decision.question,
            "deliverable": "A decision report.",
        },
        idempotency_key="test:record-delegate-1",
        skip_baseline_gate=True,
    )
    await OrchestrationAuthorityDecisionService().link_delegation_action(
        db_session, decision, action_id=delegation_action.id
    )
    artifact = Artifact(
        project_id=test_project.id,
        name="decision-report",
        artifact_type="decision_report",
        linked_task_id=delegation_action.target_id,
        created_by_agent=manager.id,
        metadata_={"selected_option": "approve", "reason": "Roster covers every work function."},
    )
    db_session.add(artifact)
    await db_session.flush()

    action = await service.execute_record_authority_decision_action(
        db_session,
        run_id=run.id,
        request={
            "action_type": "record_authority_decision",
            "decision_id": str(decision.id),
            "artifact_id": str(artifact.id),
        },
        idempotency_key="test:record-2",
    )

    assert action.status == "completed"
    await db_session.refresh(decision)
    assert decision.status == "answered"
    assert decision.selected_option == "approve"
    assert decision.decided_by_agent_id == manager.id


@pytest.mark.asyncio
async def test_execute_record_authority_decision_action_rejects_report_from_wrong_agent(db_session, test_project):
    service = OrchestrationService()
    goal, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Wrong agent report test",
            success_criteria=[{"key": "done", "description": "Done"}],
        ),
        created_by_user_id=None,
    )
    manager = Agent(
        name=f"manager-{uuid.uuid4()}", role="manager", provider="openai", model="gpt-4o-mini",
        adapter_type="api", capabilities=["coordination"], config={}, is_active=True,
    )
    impostor = Agent(
        name=f"impostor-{uuid.uuid4()}", role="worker", provider="openai", model="gpt-4o-mini",
        adapter_type="api", capabilities=[], config={}, is_active=True,
    )
    db_session.add_all([manager, impostor])
    await db_session.flush()
    decision = await OrchestrationAuthorityDecisionService().create_pending(
        db_session, goal.id, decision_key="test:record-wrong", title="Approve",
        question="Approve?", authority="manager", authority_agent_id=manager.id,
        options=["approve"], run_id=run.id,
    )
    delegation_action = await service.execute_create_delegation_task_action(
        db_session,
        run_id=run.id,
        request={
            "action_type": "create_delegation_task",
            "agent_id": str(manager.id),
            "work_function": "decision",
            "scope": decision.question,
            "deliverable": "A decision report.",
        },
        idempotency_key="test:record-wrong-delegate",
        skip_baseline_gate=True,
    )
    await OrchestrationAuthorityDecisionService().link_delegation_action(
        db_session, decision, action_id=delegation_action.id
    )
    artifact = Artifact(
        project_id=test_project.id,
        name="decision-report",
        artifact_type="decision_report",
        linked_task_id=delegation_action.target_id,
        created_by_agent=impostor.id,
        metadata_={"selected_option": "approve"},
    )
    db_session.add(artifact)
    await db_session.flush()

    with pytest.raises(HTTPException) as exc_info:
        await service.execute_record_authority_decision_action(
            db_session,
            run_id=run.id,
            request={
                "action_type": "record_authority_decision",
                "decision_id": str(decision.id),
                "artifact_id": str(artifact.id),
            },
            idempotency_key="test:record-wrong-1",
        )
    assert exc_info.value.status_code == 409
    assert "not linked to this decision's own delegation task" in exc_info.value.detail


@pytest.mark.asyncio
async def test_execute_record_authority_decision_action_rejects_report_from_other_pending_decisions_task(
    db_session, test_project
):
    """Same manager, two pending decisions with overlapping option keys
    (both offer 'approve'). A report filed against decision A's delegation
    task must not be usable to answer decision B -- only the artifact
    linked to *this* decision's own delegation task counts (cross-wiring
    guard)."""
    service = OrchestrationService()
    goal, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Cross-wiring guard test",
            success_criteria=[{"key": "done", "description": "Done"}],
        ),
        created_by_user_id=None,
    )
    manager = Agent(
        name=f"manager-{uuid.uuid4()}", role="manager", provider="openai", model="gpt-4o-mini",
        adapter_type="api", capabilities=["coordination"], config={}, is_active=True,
    )
    db_session.add(manager)
    await db_session.flush()
    decision_a = await OrchestrationAuthorityDecisionService().create_pending(
        db_session, goal.id, decision_key="test:cross-wire-a", title="Approve hierarchy",
        question="Approve hierarchy?", authority="manager", authority_agent_id=manager.id,
        options=["approve", "reject"], run_id=run.id,
    )
    decision_b = await OrchestrationAuthorityDecisionService().create_pending(
        db_session, goal.id, decision_key="test:cross-wire-b", title="Approve scope",
        question="Approve scope?", authority="manager", authority_agent_id=manager.id,
        options=["approve", "reject"], run_id=run.id,
    )
    delegation_a = await service.execute_create_delegation_task_action(
        db_session,
        run_id=run.id,
        request={
            "action_type": "create_delegation_task",
            "agent_id": str(manager.id),
            "work_function": "decision",
            "scope": decision_a.question,
            "deliverable": "A decision report.",
        },
        idempotency_key="test:cross-wire-delegate-a",
        skip_baseline_gate=True,
    )
    delegation_b = await service.execute_create_delegation_task_action(
        db_session,
        run_id=run.id,
        request={
            "action_type": "create_delegation_task",
            "agent_id": str(manager.id),
            "work_function": "decision",
            "scope": decision_b.question,
            "deliverable": "A decision report.",
        },
        idempotency_key="test:cross-wire-delegate-b",
        skip_baseline_gate=True,
    )
    authority_service = OrchestrationAuthorityDecisionService()
    await authority_service.link_delegation_action(db_session, decision_a, action_id=delegation_a.id)
    await authority_service.link_delegation_action(db_session, decision_b, action_id=delegation_b.id)
    # Report is filed against decision A's task but the LLM mistakenly (or
    # maliciously) tries to record it against decision B.
    artifact = Artifact(
        project_id=test_project.id,
        name="decision-report",
        artifact_type="decision_report",
        linked_task_id=delegation_a.target_id,
        created_by_agent=manager.id,
        metadata_={"selected_option": "approve", "reason": "Looks fine."},
    )
    db_session.add(artifact)
    await db_session.flush()

    with pytest.raises(HTTPException) as exc_info:
        await service.execute_record_authority_decision_action(
            db_session,
            run_id=run.id,
            request={
                "action_type": "record_authority_decision",
                "decision_id": str(decision_b.id),
                "artifact_id": str(artifact.id),
            },
            idempotency_key="test:cross-wire-record-b",
        )
    assert exc_info.value.status_code == 409
    assert "not linked to this decision's own delegation task" in exc_info.value.detail
    await db_session.refresh(decision_b)
    assert decision_b.status == "pending"


@pytest.mark.asyncio
async def test_execute_cancel_pending_decision_action_cancels_with_reason(db_session, test_project):
    service = OrchestrationService()
    goal, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Cancel pending decision test",
            success_criteria=[{"key": "done", "description": "Done"}],
        ),
        created_by_user_id=None,
    )
    decision = await OrchestrationAuthorityDecisionService().create_pending(
        db_session, goal.id, decision_key="test:cancel-action", title="Clarify",
        question="q", authority="human", run_id=run.id,
    )

    action = await service.execute_cancel_pending_decision_action(
        db_session,
        run_id=run.id,
        request={
            "action_type": "cancel_pending_decision",
            "decision_id": str(decision.id),
            "reason": "No longer relevant after scope change.",
        },
        idempotency_key="test:cancel-action-1",
    )

    assert action.status == "completed"
    await db_session.refresh(decision)
    assert decision.status == "cancelled"
    assert decision.reason == "No longer relevant after scope change."


@pytest.mark.asyncio
async def test_sync_agent_authority_decisions_delegates_then_records_then_escalates(db_session, test_project):
    service = OrchestrationService()
    goal, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Full round trip test",
            success_criteria=[{"key": "done", "description": "Done"}],
        ),
        created_by_user_id=None,
    )
    manager = Agent(
        name=f"manager-{uuid.uuid4()}", role="manager", provider="openai", model="gpt-4o-mini",
        adapter_type="api", capabilities=["coordination"], config={}, is_active=True,
    )
    db_session.add(manager)
    await db_session.flush()
    decision = await OrchestrationAuthorityDecisionService().create_pending(
        db_session, goal.id, decision_key="test:sync-round-trip", title="Approve hierarchy",
        question="Approve the proposed hierarchy?", authority="manager", authority_agent_id=manager.id,
        options=["approve", "request_changes"], run_id=run.id,
    )

    # Tick 1: no delegation task exists yet -> delegate.
    summary = await service._sync_agent_authority_decisions(db_session, goal, run)
    assert summary == {"delegated": 1, "recorded": 0, "escalated": 0}
    await db_session.refresh(decision)
    assert decision.related_action_id is not None
    action = await db_session.get(OrchestrationAction, decision.related_action_id)
    task = await db_session.get(Task, action.target_id)
    assert task.metadata_["orchestration"] == {
        "goal_id": str(goal.id),
        "run_id": str(run.id),
        "action_id": str(action.id),
        "work_function": "decision",
        "authority_decision_id": str(decision.id),
    }

    # Tick 2: delegation exists but no report yet -> no-op.
    summary = await service._sync_agent_authority_decisions(db_session, goal, run)
    assert summary == {"delegated": 0, "recorded": 0, "escalated": 0}
    await db_session.refresh(decision)
    assert decision.status == "pending"

    # The manager "submits" its report.
    action = await db_session.get(OrchestrationAction, decision.related_action_id)
    artifact = Artifact(
        project_id=test_project.id,
        name="decision-report",
        artifact_type="decision_report",
        linked_task_id=action.target_id,
        created_by_agent=manager.id,
        metadata_={"selected_option": "approve", "reason": "Roster is sufficient."},
    )
    db_session.add(artifact)
    await db_session.flush()

    # Tick 3: report has landed -> record.
    summary = await service._sync_agent_authority_decisions(db_session, goal, run)
    assert summary == {"delegated": 0, "recorded": 1, "escalated": 0}
    await db_session.refresh(decision)
    assert decision.status == "answered"
    assert decision.selected_option == "approve"


@pytest.mark.asyncio
async def test_sync_agent_authority_decisions_backfills_prelinked_delivery_marker(db_session, test_project):
    service = OrchestrationService()
    goal, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Backfill delivery marker test",
            success_criteria=[{"key": "done", "description": "Done"}],
        ),
        created_by_user_id=None,
    )
    manager = _agent()
    db_session.add(manager)
    await db_session.flush()
    decision = await OrchestrationAuthorityDecisionService().create_pending(
        db_session, goal.id, decision_key="test:sync-backfill", title="Approve",
        question="Approve?", authority="manager", authority_agent_id=manager.id,
        options=["approve"], run_id=run.id,
    )
    action = await service.execute_create_delegation_task_action(
        db_session,
        run_id=run.id,
        request={
            "action_type": "create_delegation_task", "agent_id": str(manager.id),
            "work_function": "decision", "scope": decision.question, "deliverable": "A decision report.",
        },
        idempotency_key="test:sync-backfill-delegate",
        skip_baseline_gate=True,
    )
    await OrchestrationAuthorityDecisionService().link_delegation_action(
        db_session, decision, action_id=action.id
    )

    assert await service._sync_agent_authority_decisions(db_session, goal, run) == {
        "delegated": 0, "recorded": 0, "escalated": 0,
    }

    task = await db_session.get(Task, action.target_id)
    assert task.metadata_["orchestration"] == {
        "goal_id": str(goal.id),
        "run_id": str(run.id),
        "action_id": str(action.id),
        "work_function": "decision",
        "authority_decision_id": str(decision.id),
    }


@pytest.mark.asyncio
async def test_sync_relinks_existing_authority_delegation_without_counting_it(db_session, test_project):
    service = OrchestrationService()
    goal, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Repair unlinked authority delegation", success_criteria=[{"key": "done", "description": "Done"}],
        ),
        created_by_user_id=None,
    )
    manager = _agent()
    db_session.add(manager)
    await db_session.flush()
    decision = await OrchestrationAuthorityDecisionService().create_pending(
        db_session, goal.id, decision_key="test:sync-relink", title="Approve", question="Approve?",
        authority="manager", authority_agent_id=manager.id, options=["approve"], run_id=run.id,
    )
    action = await service.execute_create_delegation_task_action(
        db_session,
        run_id=run.id,
        request={
            "action_type": "create_delegation_task", "agent_id": str(manager.id),
            "work_function": "decision", "scope": decision.question,
            "deliverable": "A decision report selecting one of the offered options.",
        },
        idempotency_key=f"authority_decision_delegate:{decision.id}",
        skip_baseline_gate=True,
    )

    assert await service._sync_agent_authority_decisions(db_session, goal, run) == {
        "delegated": 0, "recorded": 0, "escalated": 0,
    }
    await db_session.refresh(decision)
    assert decision.related_action_id == action.id
    actions = list(await db_session.scalars(
        select(OrchestrationAction).where(OrchestrationAction.idempotency_key == action.idempotency_key)
    ))
    assert actions == [action]
    task = await db_session.get(Task, action.target_id)
    assert task.metadata_["orchestration"]["authority_decision_id"] == str(decision.id)


@pytest.mark.asyncio
async def test_sync_agent_authority_decisions_escalates_invalid_report(db_session, test_project):
    service = OrchestrationService()
    goal, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Invalid report escalation test",
            success_criteria=[{"key": "done", "description": "Done"}],
        ),
        created_by_user_id=None,
    )
    manager = Agent(
        name=f"manager-{uuid.uuid4()}", role="manager", provider="openai", model="gpt-4o-mini",
        adapter_type="api", capabilities=["coordination"], config={}, is_active=True,
    )
    db_session.add(manager)
    await db_session.flush()
    decision = await OrchestrationAuthorityDecisionService().create_pending(
        db_session, goal.id, decision_key="test:sync-invalid-report", title="Approve",
        question="Approve?", authority="manager", authority_agent_id=manager.id,
        options=["approve", "reject"], run_id=run.id,
    )
    await service._sync_agent_authority_decisions(db_session, goal, run)
    await db_session.refresh(decision)
    action = await db_session.get(OrchestrationAction, decision.related_action_id)
    artifact = Artifact(
        project_id=test_project.id,
        name="decision-report",
        artifact_type="decision_report",
        linked_task_id=action.target_id,
        created_by_agent=manager.id,
        metadata_={"selected_option": "not_an_offered_option"},
    )
    db_session.add(artifact)
    await db_session.flush()

    summary = await service._sync_agent_authority_decisions(db_session, goal, run)

    assert summary == {"delegated": 0, "recorded": 0, "escalated": 1}
    await db_session.refresh(decision)
    assert decision.status == "cancelled"
    escalated = [
        d for d in await OrchestrationAuthorityDecisionService().list_decisions(db_session, goal.id, status="pending")
        if d.decision_key == "test:sync-invalid-report" and d.authority == "human"
    ]
    assert len(escalated) == 1


@pytest.mark.asyncio
async def test_sync_agent_authority_decisions_escalates_nulled_authority_agent(db_session, test_project):
    service = OrchestrationService()
    goal, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Nulled agent sync test",
            success_criteria=[{"key": "done", "description": "Done"}],
        ),
        created_by_user_id=None,
    )
    manager = Agent(
        name=f"manager-{uuid.uuid4()}", role="manager", provider="openai", model="gpt-4o-mini",
        adapter_type="api", capabilities=["coordination"], config={}, is_active=True,
    )
    db_session.add(manager)
    await db_session.flush()
    decision = await OrchestrationAuthorityDecisionService().create_pending(
        db_session, goal.id, decision_key="test:sync-nulled", title="Approve",
        question="Approve?", authority="manager", authority_agent_id=manager.id,
        options=["approve"], run_id=run.id,
    )
    await db_session.delete(manager)
    await db_session.flush()
    await db_session.refresh(decision)
    assert decision.authority_agent_id is None

    summary = await service._sync_agent_authority_decisions(db_session, goal, run)

    assert summary == {"delegated": 0, "recorded": 0, "escalated": 1}
    await db_session.refresh(decision)
    assert decision.status == "cancelled"


@pytest.mark.asyncio
async def test_sync_agent_authority_decisions_escalates_inactive_agent_once(db_session, test_project):
    service = OrchestrationService()
    goal, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Inactive agent sync test",
            success_criteria=[{"key": "done", "description": "Done"}],
        ),
        created_by_user_id=None,
    )
    manager = _agent()
    db_session.add(manager)
    await db_session.flush()
    decision_service = OrchestrationAuthorityDecisionService()
    delegated = await decision_service.create_pending(
        db_session, goal.id, decision_key="test:sync-inactive:delegated", title="Approve",
        question="Approve?", authority="manager", authority_agent_id=manager.id,
        options=["approve"], run_id=run.id,
    )
    assert await service._sync_agent_authority_decisions(db_session, goal, run) == {
        "delegated": 1, "recorded": 0, "escalated": 0,
    }
    undecided = await decision_service.create_pending(
        db_session, goal.id, decision_key="test:sync-inactive:undecided", title="Reject",
        question="Reject?", authority="manager", authority_agent_id=manager.id,
        options=["reject"], run_id=run.id,
    )
    manager.is_active = False
    await db_session.flush()

    assert await service._sync_agent_authority_decisions(db_session, goal, run) == {
        "delegated": 0, "recorded": 0, "escalated": 2,
    }
    await db_session.refresh(delegated)
    await db_session.refresh(undecided)
    assert delegated.status == undecided.status == "cancelled"
    pending = await decision_service.list_decisions(db_session, goal.id, status="pending")
    assert len([item for item in pending if item.authority == "human"]) == 2
    warnings = list(await db_session.scalars(
        select(OrchestrationWarning).where(
            OrchestrationWarning.goal_id == goal.id,
            OrchestrationWarning.warning_type == "agent_authority_decision_escalated",
            OrchestrationWarning.active.is_(True),
        )
    ))
    assert len(warnings) == 2
    assert {warning.related_authority_decision_id for warning in warnings} == {delegated.id, undecided.id}
    assert {warning.related_agent_id for warning in warnings} == {manager.id}

    assert await service._sync_agent_authority_decisions(db_session, goal, run) == {
        "delegated": 0, "recorded": 0, "escalated": 0,
    }
    warnings_after_retry = list(await db_session.scalars(
        select(OrchestrationWarning).where(
            OrchestrationWarning.goal_id == goal.id,
            OrchestrationWarning.warning_type == "agent_authority_decision_escalated",
            OrchestrationWarning.active.is_(True),
        )
    ))
    assert {warning.id for warning in warnings_after_retry} == {warning.id for warning in warnings}


@pytest.mark.asyncio
async def test_tick_syncs_pending_manager_decisions_and_reports_summary(db_session, test_project):
    service = OrchestrationService()
    # Empty success_criteria -> weight="trivial" -> all four baseline
    # processes complete in this single tick (verified: test_create_goal_
    # with_empty_success_criteria_asks_for_them in test_orchestration_api.py).
    goal, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(objective="Trivial tick wiring test", success_criteria=[]),
        created_by_user_id=None,
    )

    result = await service.tick(db_session, run.id)
    assert result["authority_interview"] == {"delegated": 0, "recorded": 0, "escalated": 0}

    manager = Agent(
        name=f"manager-{uuid.uuid4()}", role="manager", provider="openai", model="gpt-4o-mini",
        adapter_type="api", capabilities=["coordination"], config={}, is_active=True,
    )
    db_session.add(manager)
    await db_session.flush()
    await OrchestrationAuthorityDecisionService().create_pending(
        db_session, goal.id, decision_key="test:tick-wiring", title="Approve",
        question="Approve?", authority="manager", authority_agent_id=manager.id,
        options=["approve"], run_id=run.id,
    )

    result = await service.tick(db_session, run.id)
    assert result["authority_interview"]["delegated"] == 1


@pytest.mark.asyncio
async def test_tick_excludes_unlinked_recovered_delivery_before_marker_backfill(db_session, test_project, monkeypatch):
    from huddleroom.services.orchestration_agent_definition_review import AgentDefinitionReviewProcess

    service = OrchestrationService()
    goal, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Legacy prelinked tick test",
            success_criteria=[{"key": "done", "description": "Done"}],
        ),
        created_by_user_id=None,
    )
    run.baseline_authorized = True

    manager = _agent()
    db_session.add(manager)
    await db_session.flush()
    decision = await OrchestrationAuthorityDecisionService().create_pending(
        db_session, goal.id, decision_key="test:tick-prelinked", title="Approve",
        question="Approve?", authority="manager", authority_agent_id=manager.id,
        options=["approve"], run_id=run.id,
    )
    action = await service.execute_create_delegation_task_action(
        db_session,
        run_id=run.id,
        request={
            "action_type": "create_delegation_task", "agent_id": str(manager.id),
            "work_function": "decision", "scope": decision.question, "deliverable": "A decision report.",
        },
        idempotency_key=f"authority_decision_delegate:{decision.id}",
        skip_baseline_gate=True,
    )

    assignments_seen = []
    original_advance = AgentDefinitionReviewProcess.advance

    async def capture_assignments(process, db, tick_goal, tick_run, *, manual=False):
        assignments_seen.append(await process._current_run_assignments(db, tick_goal.project_id, tick_run.id))
        return await original_advance(process, db, tick_goal, tick_run, manual=manual)

    monkeypatch.setattr(AgentDefinitionReviewProcess, "advance", capture_assignments)

    result = await service.tick(db_session, run.id)

    assert result["authority_interview"] == {"delegated": 0, "recorded": 0, "escalated": 0}
    assert assignments_seen == [[]]
    await db_session.refresh(decision)
    assert decision.related_action_id == action.id
    task = await db_session.get(Task, action.target_id)
    assert task.metadata_["orchestration"]["authority_decision_id"] == str(decision.id)


from huddleroom.services.orchestration_process_service import OrchestrationProcessService
from huddleroom.services.orchestration_team_hierarchy import TeamHierarchyProcess


def _hier_agent(name: str, role: str, capabilities: list[str]) -> Agent:
    return Agent(
        name=name,
        role=role,
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=capabilities,
        config={"reasoning_effort": "high"},
        # A non-blank description is required: service.tick() always runs the
        # real deterministic agent_definition_review pass (it re-derives its
        # own fingerprint every tick and reruns whenever a prior process's
        # outputs don't match it), and that pass auto-disqualifies any agent
        # with a blank description/system_prompt from every work function.
        description=f"A {role} agent responsible for {', '.join(capabilities)}.",
        is_active=True,
    )


async def _build_reviewed_roster(db_session, goal, run):
    """Active agents covering every required work function plus distinct
    verifiers. Their reviews are produced for real by the deterministic
    agent_definition_review pass that service.tick() runs on its own --
    unlike TeamHierarchyProcess.advance() called directly during team hierarchy, tick()
    always reruns that pass itself, so pre-seeding OrchestrationAgentReview
    rows here would just be discarded."""
    suffix = uuid.uuid4().hex[:8]
    agents = {
        "planning": _hier_agent(f"planner-{suffix}", "planner", ["planning"]),
        "implementation": _hier_agent(f"implementer-{suffix}", "developer", ["implementation"]),
        "summarization": _hier_agent(f"summarizer-{suffix}", "writer", ["summarization"]),
        "review": _hier_agent(f"reviewer-{suffix}", "reviewer", ["review"]),
        "validation": _hier_agent(f"validator-{suffix}", "validator", ["validation"]),
    }
    db_session.add_all(agents.values())
    await db_session.flush()
    return agents


@pytest.fixture
def safe_team_hierarchy_review(monkeypatch):
    from huddleroom.services.orchestration_team_hierarchy_analyzer import (
        TeamHierarchyAnalysis,
        TeamHierarchyAnalyzer,
    )

    async def review(self, payload, project=None, *, project_id=None):
        manager_id = str(payload["selected_manager"]["id"])
        specialists = [agent for agent in payload["agents"] if str(agent["id"]) != manager_id]
        assignments = tuple(
            {"work_function": work_function, "agent_ref": next(
                str(agent["id"]) for agent in specialists if work_function in agent["capabilities"]
            )}
            for work_function in payload["required_work_functions"]
        )
        return TeamHierarchyAnalysis(
            proposed_agents=(),
            assignments=assignments,
            reporting_lines=tuple(
                {"agent_ref": assignment["agent_ref"], "reports_to": "manager"}
                for assignment in assignments
            ),
            documented_gaps=(),
            rationale="Maps each required function to the reviewed specialist roster.",
            self_review="Checked assignments and manager reporting lines.",
        )

    monkeypatch.setattr(TeamHierarchyAnalyzer, "review", review)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_manager_decision_round_trip_resumes_team_hierarchy(
    db_session, test_project, safe_team_hierarchy_review
):
    service = OrchestrationService()
    manager = Agent(
        name=f"manager-{uuid.uuid4()}", role="manager", provider="openai", model="gpt-4o-mini",
        adapter_type="api", capabilities=["coordination", "planning", "implementation", "review", "validation"],
        config={}, is_active=True,
    )
    db_session.add(manager)
    await db_session.flush()
    goal, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Implement and independently verify the export pipeline",
            success_criteria=[
                {"key": "shipped", "description": "Feature is implemented and independently verified."},
            ],
            explicit_multi_work_function=True,
        ),
        created_by_user_id=None,
    )
    run.baseline_authorized = True
    goal.weight = "substantial"
    goal.manager_agent_id = manager.id
    goal.authority_model = "agent_manager"
    process_service = OrchestrationProcessService()
    goal_def = await process_service.start_process(
        db_session, goal.id, process_type="goal_definition", trigger_reason="test setup", run_id=run.id,
    )
    await process_service.complete_process(db_session, goal_def, outputs={})
    manager_selection = await process_service.start_process(
        db_session, goal.id, process_type="manager_selection", trigger_reason="test setup", run_id=run.id,
    )
    await process_service.complete_process(db_session, manager_selection, outputs={})
    await _build_reviewed_roster(db_session, goal, run)
    await db_session.flush()

    # Tick 1: native team_hierarchy raises its nontrivial approval decision and parks.
    result = await service.tick(db_session, run.id)
    assert result["team_hierarchy_process"]["status"] == "waiting_decision"

    hierarchy_run = await process_service.get_current(db_session, goal.id, "team_hierarchy")
    decisions = await OrchestrationAuthorityDecisionService().list_decisions(db_session, goal.id, status="pending")
    approval_decision = next(d for d in decisions if d.source_process_run_id == hierarchy_run.id)

    # Test setup seeds the manager-owned decision seam rather than depending on
    # native hierarchy selection of a manager authority.
    approval_decision.authority = "manager"
    approval_decision.authority_agent_id = manager.id
    await db_session.flush()

    # Tick 2 delegates the seeded manager decision.
    result = await service.tick(db_session, run.id)
    assert result["authority_interview"]["delegated"] == 1
    delegation_action = await db_session.get(OrchestrationAction, approval_decision.related_action_id)

    # The manager "does the work" and submits its decision report.
    artifact = Artifact(
        project_id=test_project.id,
        name="hierarchy-approval-report",
        artifact_type="decision_report",
        linked_task_id=delegation_action.target_id,
        created_by_agent=manager.id,
        metadata_={"selected_option": "approve", "reason": "Roster covers every required work function."},
    )
    db_session.add(artifact)
    await db_session.flush()

    # Tick 3: the report is recorded onto the decision (still parked --
    # team_hierarchy re-checks on its own next advance() call).
    result = await service.tick(db_session, run.id)
    assert result["authority_interview"]["recorded"] == 1
    await db_session.refresh(approval_decision)
    assert approval_decision.status == "answered"

    # Tick 4: team_hierarchy's own advance() sees the answered decision,
    # verifies the manager is still active, resumes, and completes -- no
    # Authority-interview code runs here; this proves zero changes were needed to
    # orchestration_team_hierarchy.py.
    result = await service.tick(db_session, run.id)
    assert result["team_hierarchy_process"]["status"] == "completed"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_escalated_manager_decision_resumed_by_human_answer(
    db_session, test_project, test_user, safe_team_hierarchy_review
):
    """Covers the resume-from-escalation path flagged in review: once
    _sync_agent_authority_decisions escalates an invalid/undeliverable
    manager decision to a fresh human-authority row (same decision_key,
    same source_process_run_id), a human answering it through the normal
    answer endpoint must let team_hierarchy's own advance() resume --
    exercising _principal_is_current's human branch on a row that started
    life as a manager decision, per Context §11."""
    service = OrchestrationService()
    manager = Agent(
        name=f"manager-{uuid.uuid4()}", role="manager", provider="openai", model="gpt-4o-mini",
        adapter_type="api", capabilities=["coordination", "planning", "implementation", "review", "validation"],
        config={}, is_active=True,
    )
    db_session.add(manager)
    await db_session.flush()
    goal, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Implement and independently verify the export pipeline",
            success_criteria=[
                {"key": "shipped", "description": "Feature is implemented and independently verified."},
            ],
            explicit_multi_work_function=True,
        ),
        created_by_user_id=None,
    )
    run.baseline_authorized = True
    goal.weight = "substantial"
    goal.manager_agent_id = manager.id
    goal.authority_model = "agent_manager"
    process_service = OrchestrationProcessService()
    goal_def = await process_service.start_process(
        db_session, goal.id, process_type="goal_definition", trigger_reason="test setup", run_id=run.id,
    )
    await process_service.complete_process(db_session, goal_def, outputs={})
    manager_selection = await process_service.start_process(
        db_session, goal.id, process_type="manager_selection", trigger_reason="test setup", run_id=run.id,
    )
    await process_service.complete_process(db_session, manager_selection, outputs={})
    await _build_reviewed_roster(db_session, goal, run)
    await db_session.flush()

    # Tick 1: native team_hierarchy raises its nontrivial approval decision and parks.
    result = await service.tick(db_session, run.id)
    assert result["team_hierarchy_process"]["status"] == "waiting_decision"

    hierarchy_run = await process_service.get_current(db_session, goal.id, "team_hierarchy")
    decisions = await OrchestrationAuthorityDecisionService().list_decisions(db_session, goal.id, status="pending")
    approval_decision = next(d for d in decisions if d.source_process_run_id == hierarchy_run.id)

    # Test setup seeds the manager-owned decision seam rather than depending on
    # native hierarchy selection of a manager authority.
    approval_decision.authority = "manager"
    approval_decision.authority_agent_id = manager.id
    await db_session.flush()

    # Tick 2 delegates the seeded manager decision.
    result = await service.tick(db_session, run.id)
    assert result["authority_interview"]["delegated"] == 1
    delegation_action = await db_session.get(OrchestrationAction, approval_decision.related_action_id)

    # The manager submits an invalid report (bad selected_option).
    artifact = Artifact(
        project_id=test_project.id,
        name="hierarchy-approval-report",
        artifact_type="decision_report",
        linked_task_id=delegation_action.target_id,
        created_by_agent=manager.id,
        metadata_={"selected_option": "not_an_offered_option"},
    )
    db_session.add(artifact)
    await db_session.flush()

    # Tick 3: the invalid report is caught and escalated to a fresh
    # human-authority decision under the same decision_key /
    # source_process_run_id.
    result = await service.tick(db_session, run.id)
    assert result["authority_interview"]["escalated"] == 1
    await db_session.refresh(approval_decision)
    assert approval_decision.status == "cancelled"

    escalated = next(
        d for d in await OrchestrationAuthorityDecisionService().list_decisions(db_session, goal.id, status="pending")
        if d.source_process_run_id == hierarchy_run.id
    )
    assert escalated.authority == "human"

    # A human answers the escalated decision through the normal endpoint
    # path (simulated directly via the service, since this test drives the
    # service layer rather than the HTTP client). Must be a real, active
    # user -- _principal_is_current looks it up and requires is_active.
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, escalated, selected_option="approve", decided_by_user_id=test_user.id,
    )

    # Tick 4: team_hierarchy's own advance() must find the human-answered
    # row, pass _principal_is_current (human branch), and resume -- proving
    # the escalation-resume path does not deadlock.
    result = await service.tick(db_session, run.id)
    assert result["team_hierarchy_process"]["status"] == "completed"


@pytest.mark.asyncio
async def test_record_authority_decision_fails_reserved_action_when_report_missing_selected_option(db_session, test_project):
    service = OrchestrationService()
    goal, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Stuck reserved guard test",
            success_criteria=[{"key": "done", "description": "Done"}],
        ),
        created_by_user_id=None,
    )
    manager = Agent(
        name=f"manager-{uuid.uuid4()}", role="manager", provider="openai", model="gpt-4o-mini",
        adapter_type="api", capabilities=["coordination"], config={}, is_active=True,
    )
    db_session.add(manager)
    await db_session.flush()
    decision = await OrchestrationAuthorityDecisionService().create_pending(
        db_session, goal.id, decision_key="test:missing-option", title="Approve",
        question="Approve?", authority="manager", authority_agent_id=manager.id,
        options=["approve"], run_id=run.id,
    )
    delegation_action = await service.execute_create_delegation_task_action(
        db_session, run_id=run.id,
        request={
            "action_type": "create_delegation_task", "agent_id": str(manager.id),
            "work_function": "decision", "scope": decision.question, "deliverable": "A decision report.",
        },
        idempotency_key="test:missing-option-delegate", skip_baseline_gate=True,
    )
    await OrchestrationAuthorityDecisionService().link_delegation_action(
        db_session, decision, action_id=delegation_action.id
    )
    artifact = Artifact(
        project_id=test_project.id, name="decision-report", artifact_type="decision_report",
        linked_task_id=delegation_action.target_id, created_by_agent=manager.id, metadata_={},
    )
    db_session.add(artifact)
    await db_session.flush()

    with pytest.raises(HTTPException):
        await service.execute_record_authority_decision_action(
            db_session, run_id=run.id,
            request={
                "action_type": "record_authority_decision",
                "decision_id": str(decision.id), "artifact_id": str(artifact.id),
            },
            idempotency_key="test:missing-option-record",
        )
    stuck = await service._existing_action_for_key(db_session, run.id, "test:missing-option-record")
    assert stuck is not None
    assert stuck.status == "failed"
