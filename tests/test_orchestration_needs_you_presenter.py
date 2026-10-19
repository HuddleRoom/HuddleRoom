"""Acceptance coverage for deterministic Needs You reference projection.

These endpoint assertions intentionally avoid importing the new presenter until
its production module exists. Direct resolver batching coverage is added in the
green phase; the red phase must fail on missing response behavior, not imports.
"""

import uuid

import pytest
from sqlalchemy import event as sa_event

from huddleroom.models.agent import Agent
from huddleroom.models.artifact import Artifact
from huddleroom.models.meeting import Meeting
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGate, OrchestrationGoal, OrchestrationRun
from huddleroom.models.orchestration_process import OrchestrationProcessRun
from huddleroom.models.project import Project
from huddleroom.models.task import Task
from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
from huddleroom.services.orchestration_needs_you_presenter import (
    NeedsYouReferenceRequest,
    resolve_needs_you_references,
    typed_blocker_reference_requests,
)
from huddleroom.services.orchestration_warning_service import OrchestrationWarningService


def test_spr210_active_blockers_only_accept_explicit_task_and_gate_fields():
    goal_id = uuid.uuid4()
    task_id = uuid.uuid4()
    gate_id = uuid.uuid4()
    arbitrary_ids = {
        "agent_id": str(uuid.uuid4()),
        "meeting_id": str(uuid.uuid4()),
        "goal_id": str(uuid.uuid4()),
        "process_run_id": str(uuid.uuid4()),
        "action_id": str(uuid.uuid4()),
        "source_id": str(uuid.uuid4()),
    }

    assert typed_blocker_reference_requests(
        [{"task_id": str(task_id), "gate_id": str(gate_id), **arbitrary_ids}], goal_id=goal_id,
    ) == [
        NeedsYouReferenceRequest("task", task_id, goal_id),
        NeedsYouReferenceRequest("gate", gate_id, goal_id),
    ]


@pytest.mark.asyncio
async def test_spr210_resolver_is_scoped_deduplicated_and_chunk_batched(
    db_session, test_engine, test_project,
):
    goal = OrchestrationGoal(
        id=uuid.uuid4(), project_id=test_project.id, objective="Reference goal",
        success_criteria=[{"key": "done", "description": "Finished safely"}],
    )
    run = OrchestrationRun(goal_id=goal.id, status="running")
    agent = Agent(
        name=f"Reference agent {uuid.uuid4()}", role="manager", provider="openai", model="gpt-4o-mini",
        adapter_type="api", capabilities=["coordination"], config={}, is_active=True,
    )
    foreign_project = Project(name="Foreign resolver project", description="private", config={})
    db_session.add_all([goal, run, agent, foreign_project])
    await db_session.flush()

    local_task = Task(project_id=test_project.id, title="Duplicate label", status="blocked", metadata_={})
    meeting = Meeting(
        project_id=test_project.id, title="Duplicate label", meeting_type="decision", participant_agent_ids=[],
    )
    foreign_task = Task(
        project_id=foreign_project.id, title="FOREIGN SECRET TASK", status="blocked", metadata_={},
    )
    deleted_task = Task(project_id=test_project.id, title="Deleted title", status="blocked", metadata_={})
    process = OrchestrationProcessRun(
        goal_id=goal.id, run_id=run.id, process_type="authority_interview", status="waiting_decision",
        trigger_reason="reference projection", input_snapshot={}, outputs={},
    )
    gate = OrchestrationGate(
        run_id=run.id, success_criterion_key="done", gate_type="work_completed", required_evidence={},
    )
    db_session.add_all([local_task, meeting, foreign_task, deleted_task, process, gate])
    await db_session.flush()
    deleted_task_id = deleted_task.id
    await db_session.delete(deleted_task)
    action = OrchestrationAction(
        run_id=run.id, idempotency_key="spr210:resolver-action", action_type="create_delegation_task",
        request={}, target_type="task", target_id=local_task.id,
    )
    db_session.add(action)
    await db_session.flush()

    missing_task_id = uuid.uuid4()
    requests = [
        NeedsYouReferenceRequest("goal", goal.id, goal.id),
        NeedsYouReferenceRequest("process", process.id, goal.id),
        NeedsYouReferenceRequest("gate", gate.id, goal.id),
        NeedsYouReferenceRequest("action", action.id, goal.id),
        NeedsYouReferenceRequest("task", local_task.id, goal.id),
        NeedsYouReferenceRequest("meeting", meeting.id, goal.id),
        NeedsYouReferenceRequest("agent", agent.id, goal.id),
        NeedsYouReferenceRequest("task", foreign_task.id, goal.id),
        NeedsYouReferenceRequest("task", missing_task_id, goal.id),
        NeedsYouReferenceRequest("task", deleted_task_id, goal.id),
        NeedsYouReferenceRequest("task", local_task.id, goal.id),
    ]

    async def measured(candidate_requests):
        select_count = 0

        def count_selects(_conn, _cursor, statement, _parameters, _context, _executemany):
            nonlocal select_count
            if statement.lstrip().upper().startswith("SELECT"):
                select_count += 1

        sa_event.listen(test_engine.sync_engine, "before_cursor_execute", count_selects)
        try:
            return await resolve_needs_you_references(
                db_session, project_id=test_project.id, requests=candidate_requests,
            ), select_count
        finally:
            sa_event.remove(test_engine.sync_engine, "before_cursor_execute", count_selects)

    session_state = (set(db_session.new), set(db_session.dirty), set(db_session.deleted))
    resolved, one_card_queries = await measured(requests)
    repeated, many_card_queries = await measured(requests * 25)
    chunked, chunked_queries = await measured(requests + [
        NeedsYouReferenceRequest("task", uuid.UUID(int=20_000 + index), goal.id) for index in range(501)
    ])

    # Seven kind lookups plus one bounded ownership check before the global
    # agent-roster lookup; none scale with card count.
    assert one_card_queries == 8
    assert many_card_queries == one_card_queries
    assert chunked_queries == one_card_queries + 1  # only the documented 500-id task chunk adds a query
    assert len(repeated) == len(resolved)
    assert len(chunked) == len(resolved) + 501
    assert len({(reference.kind, reference.id) for reference in resolved}) == len(resolved)
    by_key = {(reference.kind, reference.id): reference for reference in resolved}
    assert by_key[("goal", goal.id)].label == goal.objective
    assert by_key[("process", process.id)].label == process.process_type
    assert by_key[("gate", gate.id)].label == "Finished safely"
    assert by_key[("action", action.id)].label == action.action_type
    assert by_key[("task", local_task.id)].label == "Duplicate label"
    assert by_key[("meeting", meeting.id)].label == "Duplicate label"
    assert by_key[("agent", agent.id)].label == agent.name
    assert by_key[("task", foreign_task.id)].label is None
    assert by_key[("task", missing_task_id)].label is None
    assert by_key[("task", deleted_task_id)].label is None
    assert all(reference.project_id == test_project.id for reference in resolved)
    assert "FOREIGN SECRET TASK" not in repr(resolved)
    assert typed_blocker_reference_requests(
        [{"task_id": "malformed", "gate_id": str(gate.id), "reason": str(uuid.uuid4())}], goal_id=goal.id,
    ) == [NeedsYouReferenceRequest("gate", gate.id, goal.id)]
    assert (set(db_session.new), set(db_session.dirty), set(db_session.deleted)) == session_state


@pytest.mark.asyncio
async def test_artifact_reference_requires_matching_review_identity_artifact_id(db_session, test_project):
    """A copied review record must not make a different artifact look reviewed."""
    goal = OrchestrationGoal(project_id=test_project.id, objective="Artifact identity")
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id, status="running")
    bound = Artifact(project_id=test_project.id, name="bound.md", artifact_type="plan", metadata_={})
    copied = Artifact(project_id=test_project.id, name="copied.md", artifact_type="plan", metadata_={})
    db_session.add_all([run, bound, copied])
    await db_session.flush()
    record = {
        "identity": {
            "project_id": str(test_project.id), "goal_id": str(goal.id), "run_id": str(run.id),
            "artifact_id": str(bound.id), "planning_task_id": str(uuid.uuid4()), "planning_version": 1,
            "winning_session_id": None, "provenance_hash": "a" * 64, "canonical_content_hash": "b" * 64,
            "renderer_version": 1, "rendered_content_hash": "c" * 64, "review_cycle": 1,
        },
        "state": "ready", "markdown": "Bound candidate",
    }
    run.plan_state = {"plan_reviews": {str(bound.id): record, str(copied.id): record}}
    await db_session.flush()

    resolved = await resolve_needs_you_references(
        db_session,
        project_id=test_project.id,
        requests=[
            NeedsYouReferenceRequest("artifact", bound.id, goal.id, run.id),
            NeedsYouReferenceRequest("artifact", copied.id, goal.id, run.id),
        ],
    )
    labels = {reference.id: reference.label for reference in resolved}
    assert labels[bound.id] == bound.name
    assert labels[copied.id] is None


@pytest.mark.asyncio
async def test_spr210_reference_projection_is_scoped_and_batched(client, db_session, test_project):
    goal = OrchestrationGoal(
        id=uuid.uuid4(),
        project_id=test_project.id,
        objective="Ship safe reference projection",
        success_criteria=[{"key": "done", "description": "Done"}],
    )
    run = OrchestrationRun(goal_id=goal.id, status="running")
    agent = Agent(
        name=f"Reference owner {uuid.uuid4()}", role="manager", provider="openai", model="gpt-4o-mini",
        adapter_type="api", capabilities=["coordination"], config={}, is_active=True,
    )
    db_session.add_all([goal, run, agent])
    await db_session.flush()

    process = OrchestrationProcessRun(
        goal_id=goal.id, run_id=run.id, process_type="authority_interview", status="waiting_decision",
        trigger_reason="acceptance test", input_snapshot={}, outputs={},
    )
    gate = OrchestrationGate(
        run_id=run.id, success_criterion_key="done", gate_type="work_completed", required_evidence={},
    )
    local_task = Task(project_id=test_project.id, title="Duplicate label", status="blocked", metadata_={})
    foreign_project = Project(name="Foreign project", description="must stay private", config={})
    db_session.add_all([process, gate, local_task, foreign_project])
    await db_session.flush()
    foreign_task = Task(
        project_id=foreign_project.id, title="FOREIGN SECRET TASK", status="blocked", metadata_={},
    )
    db_session.add(foreign_task)
    await db_session.flush()

    local_action = OrchestrationAction(
        run_id=run.id, idempotency_key="spr210:local-task", action_type="create_delegation_task",
        request={}, target_type="task", target_id=local_task.id,
    )
    foreign_action = OrchestrationAction(
        run_id=run.id, idempotency_key="spr210:foreign-task", action_type="create_delegation_task",
        request={}, target_type="task", target_id=foreign_task.id,
    )
    db_session.add_all([local_action, foreign_action])
    await db_session.flush()

    prose_only_id = uuid.uuid4()
    decisions = OrchestrationAuthorityDecisionService()
    local = await decisions.create_pending(
        db_session, goal.id, decision_key="spr210:local", title="Local references",
        question=f"Review the typed records. Ignore prose UUID {prose_only_id}.", authority="human",
        context=f"An unrelated identifier is {prose_only_id}.", run_id=run.id,
        source_process_run_id=process.id, related_gate_id=gate.id, related_action_id=local_action.id,
    )
    owner = await decisions.create_pending(
        db_session, goal.id, decision_key="spr210:owner", title="Owner reference",
        question="Owner decision", authority="manager", authority_agent_id=agent.id, run_id=run.id,
        related_action_id=foreign_action.id,
    )
    warning = await OrchestrationWarningService().create_warning(
        db_session, goal.id, warning_type="spr210_reference_warning", severity="warning",
        message="Typed warning references", run_id=run.id, source_process_run_id=process.id,
        related_gate_id=gate.id, related_action_id=local_action.id, related_agent_id=agent.id,
    )
    await db_session.flush()

    response = await client.get(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/decisions"
    )

    assert response.status_code == 200, response.text
    items = {item["id"]: item for item in response.json()}
    assert "needs_you_references" in items[str(local.id)], (
        "decision responses must add typed Needs You reference metadata"
    )
    local_refs = {(ref["kind"], ref["id"]): ref for ref in items[str(local.id)]["needs_you_references"]}
    assert local_refs[("goal", str(goal.id))]["label"] == goal.objective
    assert local_refs[("process", str(process.id))]["label"] == process.process_type
    assert local_refs[("gate", str(gate.id))]["label"] == "Done"
    assert local_refs[("action", str(local_action.id))]["label"] == local_action.action_type
    assert local_refs[("task", str(local_task.id))]["label"] == local_task.title
    assert all(ref["id"] != str(prose_only_id) for ref in local_refs.values())

    owner_refs = {(ref["kind"], ref["id"]): ref for ref in items[str(owner.id)]["needs_you_references"]}
    assert owner_refs[("agent", str(agent.id))]["label"] == agent.name
    assert owner_refs[("task", str(foreign_task.id))]["label"] is None
    assert "FOREIGN SECRET TASK" not in response.text

    warning_response = await client.get(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/warnings"
    )
    assert warning_response.status_code == 200, warning_response.text
    warning_item = next(item for item in warning_response.json() if item["id"] == str(warning.id))
    warning_refs = {(ref["kind"], ref["id"]): ref for ref in warning_item["needs_you_references"]}
    assert warning_refs[("goal", str(goal.id))]["label"] == goal.objective
    assert warning_refs[("process", str(process.id))]["label"] == process.process_type
    assert warning_refs[("gate", str(gate.id))]["label"] == "Done"
    assert warning_refs[("action", str(local_action.id))]["label"] == local_action.action_type
    assert warning_refs[("agent", str(agent.id))]["label"] == agent.name
    assert warning_refs[("task", str(local_task.id))]["label"] == local_task.title
