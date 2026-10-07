import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import select

pytestmark = pytest.mark.usefixtures("safe_goal_analysis")

from huddleroom.models.agent import Agent
from huddleroom.models.event_log import EventLog
from huddleroom.models.meeting import Meeting, MeetingAgendaItem, MeetingDecision
from huddleroom.models.orchestration import OrchestrationEvidence, OrchestrationGate
from huddleroom.models.graph import Graph, GraphRun
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.schemas.orchestration import OrchestrationGoalCreate
from huddleroom.services.event_bus import emit_event_once
from huddleroom.services.orchestration_service import OrchestrationService


def _agent(name_prefix: str, role: str, capabilities: list[str]) -> Agent:
    return Agent(
        name=f"{name_prefix}-{uuid.uuid4()}",
        role=role,
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=capabilities,
        config={},
        is_active=True,
    )


async def _make_run(db_session, project_id):
    service = OrchestrationService()
    goal, run = await service.create_goal(
        db_session,
        project_id=project_id,
        data=OrchestrationGoalCreate(
            objective="Ship evidence ingestion",
            success_criteria=[{"key": "work", "description": "Runtime evidence is ingested."}],
            constraints={},
            budget={},
        ),
        created_by_user_id=None,
    )
    return service, goal, run


async def _make_gate(db_session, run_id, gate_type: str = "work_completed") -> OrchestrationGate:
    gate = OrchestrationGate(
        run_id=run_id,
        success_criterion_key="plan_item:implement-api",
        gate_type=gate_type,
        required_evidence={
            "required_source_types": ["task", "session", "review", "graph_run", "meeting_decision"],
            "min_count": 1,
            "plan_item_id": "implement-api",
        },
    )
    db_session.add(gate)
    await db_session.flush()
    return gate


async def _make_orchestrated_task(
    db_session,
    project_id,
    run_id,
    gate_id,
    agent_id,
    work_function: str = "implementation",
) -> Task:
    task = Task(
        project_id=project_id,
        title=f"{work_function.title()} work",
        description="Work created from an accepted orchestration plan item.",
        status="in_progress",
        assigned_to=agent_id,
        metadata_={
            "orchestration": {
                "goal_id": "unused-by-evidence-ingestion",
                "run_id": str(run_id),
                "work_function": work_function,
                "plan_item_id": "implement-api",
                "plan_item_gate_id": str(gate_id),
                "expand_action_id": str(uuid.uuid4()),
            },
            "orchestration_plan_item": {
                "id": "implement-api",
                "work_function": work_function,
                "accepted_plan_artifact_id": str(uuid.uuid4()),
            },
        },
    )
    db_session.add(task)
    await db_session.flush()
    return task


async def _evidence_rows(db_session, run_id):
    result = await db_session.execute(
        select(OrchestrationEvidence)
        .where(OrchestrationEvidence.run_id == run_id)
        .order_by(OrchestrationEvidence.created_at.asc(), OrchestrationEvidence.id.asc())
    )
    return list(result.scalars().all())


async def _event_log(db_session, event_id):
    result = await db_session.execute(select(EventLog).where(EventLog.id == event_id))
    return result.scalar_one()


@pytest.mark.asyncio
async def test_tick_ingests_completed_task_evidence_once(db_session, test_project):
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)
    developer = _agent("developer", "developer", ["implementation"])
    db_session.add(developer)
    await db_session.flush()
    task = await _make_orchestrated_task(db_session, test_project.id, run.id, gate.id, developer.id)
    task.status = "done"
    task.completed_at = datetime.now(timezone.utc)
    event, _ = await emit_event_once(
        db_session,
        test_project.id,
        "task.status_changed",
        {
            "task_id": str(task.id),
            "status": "done",
            "previous_status": "in_progress",
            "project_id": str(test_project.id),
        },
        dedup_key=f"phase13-task-done:{task.id}",
    )

    first = await service.tick(db_session, run.id)
    event_log = await _event_log(db_session, event.id)
    run.event_cursor = event_log.seq - 1
    await db_session.flush()
    second = await service.tick(db_session, run.id)

    rows = await _evidence_rows(db_session, run.id)
    assert first["processed_events"] == 1
    assert first["evidence_created"] == 1
    assert second["evidence_created"] == 0
    assert len(rows) == 1
    assert rows[0].gate_id == gate.id
    assert rows[0].source_type == "task"
    assert rows[0].source_id == task.id
    assert rows[0].observed_event_id == event.id
    assert rows[0].producer_agent_id == developer.id
    assert rows[0].verdict == "candidate"
    assert rows[0].evidence_metadata["event_type"] == "task.status_changed"
    assert rows[0].evidence_metadata["source_status"] == "done"
    assert rows[0].evidence_metadata["plan_item_id"] == "implement-api"


@pytest.mark.asyncio
async def test_tick_ignores_completed_task_without_orchestration_metadata(db_session, test_project):
    service, _goal, run = await _make_run(db_session, test_project.id)
    plain_task = Task(project_id=test_project.id, title="Plain task", status="done", metadata_={})
    db_session.add(plain_task)
    await db_session.flush()
    await emit_event_once(
        db_session,
        test_project.id,
        "task.status_changed",
        {
            "task_id": str(plain_task.id),
            "status": "done",
            "previous_status": "in_progress",
            "project_id": str(test_project.id),
        },
        dedup_key=f"phase13-plain-task:{plain_task.id}",
    )

    result = await service.tick(db_session, run.id)

    assert result["evidence_created"] == 0
    assert await _evidence_rows(db_session, run.id) == []


@pytest.mark.asyncio
async def test_tick_ingests_completed_session_evidence(db_session, test_project):
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)
    developer = _agent("developer", "developer", ["implementation"])
    db_session.add(developer)
    await db_session.flush()
    task = await _make_orchestrated_task(db_session, test_project.id, run.id, gate.id, developer.id)
    session = Session(
        project_id=test_project.id,
        task_id=task.id,
        agent_id=developer.id,
        adapter_type="api",
        status="completed",
        input_context={},
        output="Implemented the API and ran focused tests.",
        metadata_={"token_count_out": 120},
        origin="auto",
    )
    db_session.add(session)
    await db_session.flush()
    event, _ = await emit_event_once(
        db_session,
        test_project.id,
        "session.completed",
        {
            "session_id": str(session.id),
            "task_id": str(task.id),
            "project_id": str(test_project.id),
        },
        dedup_key=f"phase13-session-completed:{session.id}",
    )

    result = await service.tick(db_session, run.id)

    rows = await _evidence_rows(db_session, run.id)
    assert result["evidence_created"] == 1
    assert len(rows) == 1
    assert rows[0].source_type == "session"
    assert rows[0].source_id == session.id
    assert rows[0].observed_event_id == event.id
    assert rows[0].producer_agent_id == developer.id
    assert rows[0].verdict == "candidate"
    assert rows[0].evidence_metadata["session_id"] == str(session.id)
    assert rows[0].evidence_metadata["origin"] == "auto"


@pytest.mark.asyncio
async def test_tick_ingests_review_approved_evidence(db_session, test_project):
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id, gate_type="review_accepted")
    reviewer = _agent("reviewer", "reviewer", ["review"])
    db_session.add(reviewer)
    await db_session.flush()
    task = await _make_orchestrated_task(db_session, test_project.id, run.id, gate.id, reviewer.id, "review")
    session = Session(
        project_id=test_project.id,
        task_id=task.id,
        agent_id=reviewer.id,
        adapter_type="api",
        status="completed",
        input_context={},
        output="APPROVED",
        metadata_={},
        origin="graph",
    )
    db_session.add(session)
    await db_session.flush()
    event, _ = await emit_event_once(
        db_session,
        test_project.id,
        "review.approved",
        {
            "session_id": str(session.id),
            "task_id": str(task.id),
            "review_outcome": {"verdict": "approved", "raw_output": "APPROVED"},
            "project_id": str(test_project.id),
        },
        dedup_key=f"phase13-review-approved:{session.id}",
    )

    await service.tick(db_session, run.id)

    rows = await _evidence_rows(db_session, run.id)
    assert len(rows) == 1
    assert rows[0].source_type == "review"
    assert rows[0].source_id == session.id
    assert rows[0].observed_event_id == event.id
    assert rows[0].producer_agent_id == reviewer.id
    assert rows[0].verdict == "candidate"
    assert rows[0].evidence_metadata["review_verdict"] == "approved"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "violation_type",
    ["missing_session", "mismatched_session_task", "mismatched_graph_run"],
)
async def test_tick_ignores_review_with_violated_link(db_session, test_project, violation_type):
    service, _goal, run = await _make_run(db_session, test_project.id)
    reviewer = _agent("reviewer", "reviewer", ["review"])
    db_session.add(reviewer)
    await db_session.flush()

    if violation_type == "missing_session":
        gate = await _make_gate(db_session, run.id, gate_type="review_accepted")
        task = await _make_orchestrated_task(db_session, test_project.id, run.id, gate.id, reviewer.id, "review")
        await emit_event_once(
            db_session,
            test_project.id,
            "review.approved",
            {
                "session_id": str(uuid.uuid4()),
                "task_id": str(task.id),
                "review_outcome": {"verdict": "approved"},
                "project_id": str(test_project.id),
            },
            dedup_key=f"phase13-review-missing-session:{task.id}",
        )
    elif violation_type == "mismatched_session_task":
        gate_a = await _make_gate(db_session, run.id, gate_type="review_accepted")
        gate_b = await _make_gate(db_session, run.id, gate_type="review_accepted")
        task_a = await _make_orchestrated_task(db_session, test_project.id, run.id, gate_a.id, reviewer.id, "review")
        task_b = await _make_orchestrated_task(db_session, test_project.id, run.id, gate_b.id, reviewer.id, "review")
        session = Session(
            project_id=test_project.id,
            task_id=task_a.id,
            agent_id=reviewer.id,
            adapter_type="api",
            status="completed",
            input_context={},
            output="APPROVED",
            metadata_={},
            origin="graph",
        )
        db_session.add(session)
        await db_session.flush()
        await emit_event_once(
            db_session,
            test_project.id,
            "review.approved",
            {
                "session_id": str(session.id),
                "task_id": str(task_b.id),
                "review_outcome": {"verdict": "approved"},
                "project_id": str(test_project.id),
            },
            dedup_key=f"phase13-review-mismatched-task:{session.id}",
        )
    else:  # mismatched_graph_run
        gate = await _make_gate(db_session, run.id, gate_type="review_accepted")
        task = await _make_orchestrated_task(db_session, test_project.id, run.id, gate.id, reviewer.id, "review")
        graph = Graph(
            project_id=test_project.id,
            name=f"evidence-review-graph-{uuid.uuid4()}",
            version="1.0",
            definition={},
            triggers=[],
        )
        db_session.add(graph)
        await db_session.flush()
        run_a = GraphRun(
            graph_id=graph.id,
            project_id=test_project.id,
            linked_task_id=task.id,
            current_node="reviewed",
            status="completed",
            context={},
        )
        run_b = GraphRun(
            graph_id=graph.id,
            project_id=test_project.id,
            linked_task_id=task.id,
            current_node="other",
            status="completed",
            context={},
        )
        db_session.add_all([run_a, run_b])
        await db_session.flush()
        session = Session(
            project_id=test_project.id,
            task_id=task.id,
            agent_id=reviewer.id,
            graph_run_id=run_a.id,
            adapter_type="api",
            status="completed",
            input_context={},
            output="APPROVED",
            metadata_={},
            origin="graph",
        )
        db_session.add(session)
        await db_session.flush()
        await emit_event_once(
            db_session,
            test_project.id,
            "review.approved",
            {
                "session_id": str(session.id),
                "task_id": str(task.id),
                "graph_run_id": str(run_b.id),
                "review_outcome": {"verdict": "approved"},
                "project_id": str(test_project.id),
            },
            dedup_key=f"phase13-review-mismatched-graph:{session.id}",
        )

    result = await service.tick(db_session, run.id)

    assert result["evidence_created"] == 0
    assert await _evidence_rows(db_session, run.id) == []


@pytest.mark.asyncio
async def test_tick_ingests_graph_completed_evidence(db_session, test_project):
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id, gate_type="validation_passed")
    validator = _agent("validator", "validator", ["validation"])
    db_session.add(validator)
    await db_session.flush()
    task = await _make_orchestrated_task(db_session, test_project.id, run.id, gate.id, validator.id, "validation")
    graph = Graph(
        project_id=test_project.id,
        name=f"evidence-validation-{uuid.uuid4()}",
        version="1.0",
        definition={},
        triggers=[],
    )
    db_session.add(graph)
    await db_session.flush()
    graph_run = GraphRun(
        graph_id=graph.id,
        project_id=test_project.id,
        linked_task_id=task.id,
        current_node="passed",
        status="completed",
        context={},
    )
    db_session.add(graph_run)
    await db_session.flush()
    event, _ = await emit_event_once(
        db_session,
        test_project.id,
        "graph.run_completed",
        {
            "graph_run_id": str(graph_run.id),
            "graph_name": graph.name,
            "project_id": str(test_project.id),
        },
        dedup_key=f"phase13-graph-completed:{graph_run.id}",
    )

    await service.tick(db_session, run.id)

    rows = await _evidence_rows(db_session, run.id)
    assert len(rows) == 1
    assert rows[0].source_type == "graph_run"
    assert rows[0].source_id == graph_run.id
    assert rows[0].observed_event_id == event.id
    assert rows[0].verdict == "candidate"
    assert rows[0].evidence_metadata["graph_run_id"] == str(graph_run.id)
    assert rows[0].evidence_metadata["source_status"] == "completed"


@pytest.mark.asyncio
async def test_tick_ingests_meeting_decision_evidence_from_concluded_meeting(db_session, test_project):
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id, gate_type="meeting_decision_accepted")
    facilitator = _agent("facilitator", "facilitator", ["meeting"])
    db_session.add(facilitator)
    await db_session.flush()
    task = await _make_orchestrated_task(db_session, test_project.id, run.id, gate.id, facilitator.id, "meeting")
    meeting = Meeting(
        project_id=test_project.id,
        title="Resolve validation disagreement",
        meeting_type="decision",
        status="concluded",
        participant_agent_ids=[],
        participant_user_ids=[],
        source_task_id=task.id,
        organizer_agent_id=facilitator.id,
    )
    db_session.add(meeting)
    await db_session.flush()
    agenda_item = MeetingAgendaItem(meeting_id=meeting.id, order=1, title="Accept validation result")
    db_session.add(agenda_item)
    await db_session.flush()
    decision = MeetingDecision(
        meeting_id=meeting.id,
        agenda_item_id=agenda_item.id,
        title="Accept validation result",
        question="Does this satisfy the gate?",
        chosen_option="Accept",
        rationale="The independent validation evidence is sufficient.",
        decided_by="consensus",
    )
    db_session.add(decision)
    await db_session.flush()
    event, _ = await emit_event_once(
        db_session,
        test_project.id,
        "meeting.concluded",
        {
            "meeting_id": str(meeting.id),
            "project_id": str(test_project.id),
        },
        dedup_key=f"phase13-meeting-concluded:{meeting.id}",
    )

    await service.tick(db_session, run.id)

    rows = await _evidence_rows(db_session, run.id)
    assert len(rows) == 1
    assert rows[0].source_type == "meeting_decision"
    assert rows[0].source_id == decision.id
    assert rows[0].observed_event_id == event.id
    assert rows[0].producer_agent_id == facilitator.id
    assert rows[0].verdict == "candidate"
    assert rows[0].evidence_metadata["meeting_id"] == str(meeting.id)
    assert rows[0].evidence_metadata["decision_id"] == str(decision.id)


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch_type", ["different_meeting", "wrong_gate"])
async def test_tick_ignores_meeting_decision_with_violated_link(db_session, test_project, mismatch_type):
    service, _goal, run = await _make_run(db_session, test_project.id)
    facilitator = _agent("facilitator", "facilitator", ["meeting"])
    db_session.add(facilitator)
    await db_session.flush()

    if mismatch_type == "different_meeting":
        gate = await _make_gate(db_session, run.id, gate_type="meeting_decision_accepted")
        meeting_a = Meeting(
            project_id=test_project.id,
            title="Payload meeting",
            meeting_type="decision",
            status="active",
            participant_agent_ids=[],
            participant_user_ids=[],
            organizer_agent_id=facilitator.id,
        )
        meeting_b = Meeting(
            project_id=test_project.id,
            title="Decision meeting",
            meeting_type="decision",
            status="active",
            participant_agent_ids=[],
            participant_user_ids=[],
            organizer_agent_id=facilitator.id,
        )
        db_session.add_all([meeting_a, meeting_b])
        await db_session.flush()
        agenda_item = MeetingAgendaItem(meeting_id=meeting_b.id, order=1, title="Choose path")
        db_session.add(agenda_item)
        await db_session.flush()
        decision = MeetingDecision(
            meeting_id=meeting_b.id,
            agenda_item_id=agenda_item.id,
            title="Choose path",
            question="Which path should evidence use?",
            chosen_option="B",
            rationale="Decision belongs to meeting B.",
            decided_by="facilitator",
        )
        db_session.add(decision)
        await db_session.flush()
        await emit_event_once(
            db_session,
            test_project.id,
            "meeting.decision_recorded",
            {
                "meeting_id": str(meeting_a.id),
                "decision_id": str(decision.id),
                "gate_id": str(gate.id),
                "project_id": str(test_project.id),
            },
            dedup_key=f"phase13-meeting-decision-mismatch:{decision.id}",
        )
    else:  # wrong_gate
        gate_a = await _make_gate(db_session, run.id, gate_type="meeting_decision_accepted")
        gate_b = await _make_gate(db_session, run.id, gate_type="meeting_decision_accepted")
        task = await _make_orchestrated_task(db_session, test_project.id, run.id, gate_a.id, facilitator.id, "meeting")
        meeting = Meeting(
            project_id=test_project.id,
            title="Payload meeting",
            meeting_type="decision",
            status="active",
            participant_agent_ids=[],
            participant_user_ids=[],
            source_task_id=task.id,
            organizer_agent_id=facilitator.id,
        )
        db_session.add(meeting)
        await db_session.flush()
        agenda_item = MeetingAgendaItem(meeting_id=meeting.id, order=1, title="Choose path")
        db_session.add(agenda_item)
        await db_session.flush()
        decision = MeetingDecision(
            meeting_id=meeting.id,
            agenda_item_id=agenda_item.id,
            title="Choose path",
            question="Which path should evidence use?",
            chosen_option="A",
            rationale="Decision belongs to gate A's task.",
            decided_by="facilitator",
        )
        db_session.add(decision)
        await db_session.flush()
        await emit_event_once(
            db_session,
            test_project.id,
            "meeting.decision_recorded",
            {
                "meeting_id": str(meeting.id),
                "decision_id": str(decision.id),
                "gate_id": str(gate_b.id),
                "project_id": str(test_project.id),
            },
            dedup_key=f"phase13-meeting-wrong-gate:{decision.id}",
        )

    result = await service.tick(db_session, run.id)

    assert result["evidence_created"] == 0
    assert await _evidence_rows(db_session, run.id) == []


@pytest.mark.asyncio
async def test_goal_detail_lists_gates_and_ingested_evidence(client, db_session, test_project):
    service, goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)
    developer = _agent("developer", "developer", ["implementation"])
    db_session.add(developer)
    await db_session.flush()
    task = await _make_orchestrated_task(db_session, test_project.id, run.id, gate.id, developer.id)
    task.status = "done"
    await emit_event_once(
        db_session,
        test_project.id,
        "task.status_changed",
        {
            "task_id": str(task.id),
            "status": "done",
            "previous_status": "in_progress",
            "project_id": str(test_project.id),
        },
        dedup_key=f"phase13-detail-task:{task.id}",
    )
    await service.tick(db_session, run.id)

    response = await client.get(f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["gates_count"] == 1
    assert body["evidence_count"] == 1
    assert body["gates"][0]["id"] == str(gate.id)
    assert body["evidence"][0]["source_type"] == "task"
    assert body["evidence"][0]["source_id"] == str(task.id)
    assert body["evidence"][0]["observed_event_id"] is not None
