import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from fastapi import HTTPException
from sqlalchemy import func, select

pytestmark = pytest.mark.usefixtures("safe_goal_analysis")

from huddleroom.models.agent import Agent
from huddleroom.models.event_log import EventLog
from huddleroom.models.meeting import Meeting, MeetingAgendaItem, MeetingDecision
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationEvidence, OrchestrationGate
from huddleroom.models.graph import Graph, GraphRun, GraphRunTimeout, GraphRunStep
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.schemas.orchestration import OrchestrationGoalCreate
from huddleroom.services.event_bus import emit_event_once
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.graph_engine import GraphEngineService


@pytest_asyncio.fixture(autouse=True)
async def _runnable_graph_project(db_session, test_project, tmp_path):
    workspace = tmp_path / "graph-workspace"
    workspace.mkdir()
    test_project.workspace_path = str(workspace.resolve())
    await db_session.flush()


def _agent(name: str, role: str, capabilities: list[str], *, is_active: bool = True) -> Agent:
    return Agent(
        name=f"{name}-{uuid.uuid4()}",
        role=role,
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=capabilities,
        config={},
        is_active=is_active,
    )


async def _make_run(db_session, project_id):
    service = OrchestrationService()
    goal, run = await service.create_goal(
        db_session,
        project_id=project_id,
        data=OrchestrationGoalCreate(
            objective="Resolve work through existing coordination primitives",
            success_criteria=[
                {
                    "key": "coordination-complete",
                    "description": "Coordination produces accepted evidence.",
                }
            ],
            constraints={},
            budget={},
        ),
        created_by_user_id=None,
    )
    return service, goal, run


async def _make_gate(db_session, run_id, required_source_type: str) -> OrchestrationGate:
    gate = OrchestrationGate(
        run_id=run_id,
        success_criterion_key="coordination-complete",
        gate_type=(
            "meeting_decision_accepted"
            if required_source_type == "meeting_decision"
            else "validation_passed"
        ),
        required_evidence={
            "required_source_types": [required_source_type],
            "min_count": 1,
        },
        status="open",
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
    *,
    status: str = "blocked",
) -> Task:
    task = Task(
        project_id=project_id,
        title="Resolve coordination blocker",
        description="Work linked to a meeting graph gate.",
        status=status,
        assigned_to=agent_id,
        started_at=datetime.now(timezone.utc),
        metadata_={
            "orchestration": {
                "goal_id": "meeting-graph-goal",
                "run_id": str(run_id),
                "action_id": str(uuid.uuid4()),
                "work_function": "validation",
                "plan_item_id": "coordinate",
                "plan_item_gate_id": str(gate_id),
                "expand_action_id": str(uuid.uuid4()),
            },
            "orchestration_plan_item": {
                "id": "coordinate",
                "work_function": "validation",
                "accepted_plan_artifact_id": str(uuid.uuid4()),
            },
        },
    )
    db_session.add(task)
    await db_session.flush()
    return task


async def _actions(db_session, run_id, action_type: str) -> list[OrchestrationAction]:
    result = await db_session.execute(
        select(OrchestrationAction)
        .where(
            OrchestrationAction.run_id == run_id,
            OrchestrationAction.action_type == action_type,
        )
        .order_by(OrchestrationAction.created_at.asc(), OrchestrationAction.id.asc())
    )
    return list(result.scalars().all())


async def _events(db_session, project_id, event_type: str) -> list[EventLog]:
    result = await db_session.execute(
        select(EventLog)
        .where(EventLog.project_id == project_id, EventLog.event_type == event_type)
        .order_by(EventLog.seq.asc())
    )
    return list(result.scalars().all())


@pytest.mark.asyncio
async def test_schedule_meeting_action_links_task_gate_and_replays_once(db_session, test_project):
    facilitator = _agent("facilitator", "facilitator", ["meeting_facilitation"])
    reviewer = _agent("reviewer", "reviewer", ["review"])
    db_session.add_all([facilitator, reviewer])
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id, "meeting_decision")
    task = await _make_orchestrated_task(
        db_session,
        test_project.id,
        run.id,
        gate.id,
        reviewer.id,
    )
    request = {
        "action_type": "schedule_meeting",
        "topic": "Resolve contradictory validation evidence.",
        "participant_agent_ids": [str(facilitator.id), str(reviewer.id), str(facilitator.id)],
        "task_id": str(task.id),
        "gate_id": str(gate.id),
        "organizer_agent_id": str(facilitator.id),
    }
    key = f"run:{run.id}:kind:schedule_meeting:gate:{gate.id}"

    first = await service.execute_schedule_meeting_action(
        db_session,
        run_id=run.id,
        request=request,
        idempotency_key=key,
    )
    second = await service.execute_schedule_meeting_action(
        db_session,
        run_id=run.id,
        request=request,
        idempotency_key=key,
    )

    assert first.id == second.id
    assert first.status == "completed"
    assert first.target_type == "meeting"
    meeting = await db_session.get(Meeting, first.target_id)
    assert meeting is not None
    assert meeting.source_task_id == task.id
    assert meeting.organizer_agent_id == facilitator.id
    assert meeting.participant_agent_ids == [str(facilitator.id), str(reviewer.id)]
    agenda = list(
        (
            await db_session.execute(
                select(MeetingAgendaItem).where(MeetingAgendaItem.meeting_id == meeting.id)
            )
        ).scalars().all()
    )
    assert len(agenda) == 1
    assert agenda[0].title == request["topic"]
    assert await db_session.scalar(
        select(func.count(Meeting.id)).where(Meeting.source_task_id == task.id)
    ) == 1
    assert len(await _events(db_session, test_project.id, "meeting.scheduled")) == 1
    orchestration_events = await _events(
        db_session,
        test_project.id,
        "orchestration.meeting_scheduled",
    )
    assert len(orchestration_events) == 1
    assert orchestration_events[0].payload["run_id"] == str(run.id)
    assert orchestration_events[0].payload["gate_id"] == str(gate.id)


@pytest.mark.asyncio
async def test_schedule_meeting_action_fails_for_inactive_participant(db_session, test_project):
    inactive = _agent("inactive", "facilitator", ["meeting_facilitation"], is_active=False)
    db_session.add(inactive)
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id, "meeting_decision")
    task = await _make_orchestrated_task(
        db_session,
        test_project.id,
        run.id,
        gate.id,
        inactive.id,
    )

    with pytest.raises(HTTPException) as exc:
        await service.execute_schedule_meeting_action(
            db_session,
            run_id=run.id,
            request={
                "action_type": "schedule_meeting",
                "topic": "Resolve the blocker.",
                "participant_agent_ids": [str(inactive.id)],
                "task_id": str(task.id),
                "gate_id": str(gate.id),
            },
            idempotency_key=f"run:{run.id}:kind:schedule_meeting:gate:{gate.id}",
        )

    assert exc.value.status_code == 409
    assert await db_session.scalar(
        select(func.count(Meeting.id)).where(Meeting.source_task_id == task.id)
    ) == 0
    actions = await _actions(db_session, run.id, "schedule_meeting")
    assert len(actions) == 1
    assert actions[0].status == "failed"
    assert actions[0].error == "Meeting participants must be active agents"


@pytest.mark.asyncio
async def test_schedule_meeting_action_reuses_existing_active_task_meeting(db_session, test_project):
    facilitator = _agent("facilitator", "facilitator", ["meeting_facilitation"])
    db_session.add(facilitator)
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id, "meeting_decision")
    task = await _make_orchestrated_task(
        db_session,
        test_project.id,
        run.id,
        gate.id,
        facilitator.id,
    )
    existing_meeting = Meeting(
        project_id=test_project.id,
        title="Existing blocker meeting",
        meeting_type="adhoc",
        status="scheduled",
        participant_agent_ids=[str(facilitator.id)],
        participant_user_ids=[],
        source_task_id=task.id,
        organizer_agent_id=facilitator.id,
    )
    db_session.add(existing_meeting)
    await db_session.flush()

    action = await service.execute_schedule_meeting_action(
        db_session,
        run_id=run.id,
        request={
            "action_type": "schedule_meeting",
            "topic": "Resolve the blocker.",
            "participant_agent_ids": [str(facilitator.id)],
            "task_id": str(task.id),
        },
        idempotency_key=f"run:{run.id}:kind:schedule_meeting:gate:{gate.id}",
    )

    assert action.target_id == existing_meeting.id
    assert await db_session.scalar(
        select(func.count(Meeting.id)).where(Meeting.source_task_id == task.id)
    ) == 1
    assert await _events(db_session, test_project.id, "meeting.scheduled") == []
    assert len(
        await _events(db_session, test_project.id, "orchestration.meeting_scheduled")
    ) == 1


@pytest.mark.asyncio
async def test_schedule_meeting_action_respects_paused_run(db_session, test_project):
    facilitator = _agent("facilitator", "facilitator", ["meeting_facilitation"])
    db_session.add(facilitator)
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id, "meeting_decision")
    task = await _make_orchestrated_task(
        db_session,
        test_project.id,
        run.id,
        gate.id,
        facilitator.id,
    )
    run.status = "paused"
    await db_session.flush()

    with pytest.raises(HTTPException) as exc:
        await service.execute_schedule_meeting_action(
            db_session,
            run_id=run.id,
            request={
                "action_type": "schedule_meeting",
                "topic": "Resolve the blocker.",
                "participant_agent_ids": [str(facilitator.id)],
                "task_id": str(task.id),
            },
            idempotency_key=f"run:{run.id}:kind:schedule_meeting:gate:{gate.id}",
        )

    assert exc.value.status_code == 409
    assert await db_session.scalar(
        select(func.count(Meeting.id)).where(Meeting.source_task_id == task.id)
    ) == 0


async def _graph(db_session, project_id) -> Graph:
    graph = Graph(
        project_id=project_id,
        name=f"meeting-graph-review-{uuid.uuid4()}",
        version="1.0",
        definition={
            "start_node": "pending",
            "nodes": {
                "pending": {
                    "timeout": {"duration": "1h", "action": "escalate"},
                    "edges": [],
                },
                "reviewed": {"edges": []},
            },
            "terminal_nodes": {"success": [], "failure": []},
        },
        triggers=[],
        is_active=True,
    )
    db_session.add(graph)
    await db_session.flush()
    return graph


@pytest.mark.asyncio
async def test_start_graph_action_uses_engine_and_replays_once(db_session, test_project):
    validator = _agent("validator", "validator", ["validation"])
    db_session.add(validator)
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id, "graph_run_step")
    task = await _make_orchestrated_task(
        db_session,
        test_project.id,
        run.id,
        gate.id,
        validator.id,
        status="in_progress",
    )
    graph = await _graph(db_session, test_project.id)
    request = {
        "action_type": "start_graph",
        "graph_id": str(graph.id),
        "subject_type": "task",
        "subject_id": str(task.id),
    }
    key = f"run:{run.id}:kind:start_graph:graph:{graph.id}:task:{task.id}"

    first = await service.execute_start_graph_action(
        db_session,
        run_id=run.id,
        request=request,
        idempotency_key=key,
    )
    second = await service.execute_start_graph_action(
        db_session,
        run_id=run.id,
        request=request,
        idempotency_key=key,
    )

    assert first.id == second.id
    assert first.status == "completed"
    assert first.target_type == "graph_run"
    run_instance = await db_session.get(GraphRun, first.target_id)
    assert run_instance is not None
    assert run_instance.graph_id == graph.id
    assert run_instance.linked_task_id == task.id
    assert run_instance.current_node == "pending"
    assert run_instance.triggering_event_id == first.id
    steps = list(
        (
            await db_session.execute(
                select(GraphRunStep).where(GraphRunStep.graph_run_id == run_instance.id)
            )
        ).scalars().all()
    )
    assert len(steps) == 1
    assert steps[0].from_node == ""
    assert steps[0].to_node == "pending"
    assert await db_session.scalar(
        select(func.count(GraphRunTimeout.id)).where(GraphRunTimeout.graph_run_id == run_instance.id)
    ) == 1
    assert await db_session.scalar(select(func.count(GraphRun.id))) == 1
    assert len(await _events(db_session, test_project.id, "graph.run_started")) == 1
    orchestration_events = await _events(
        db_session,
        test_project.id,
        "orchestration.graph_started",
    )
    assert len(orchestration_events) == 1
    assert orchestration_events[0].payload["gate_id"] == str(gate.id)
    assert await db_session.scalar(
        select(func.count(OrchestrationEvidence.id)).where(OrchestrationEvidence.run_id == run.id)
    ) == 0


@pytest.mark.asyncio
async def test_start_graph_action_rejects_unrunnable_project_before_reservation(db_session, test_project):
    """Moving the guard after reservation would persist a failed action for a rejected execution."""
    validator = _agent("validator", "validator", ["validation"])
    db_session.add(validator)
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, test_project.id)
    test_project.workspace_path = None
    await db_session.flush()
    gate = await _make_gate(db_session, run.id, "graph_run_step")
    task = await _make_orchestrated_task(
        db_session,
        test_project.id,
        run.id,
        gate.id,
        validator.id,
        status="in_progress",
    )
    graph = await _graph(db_session, test_project.id)
    counts_before = {
        model: await db_session.scalar(select(func.count(model.id)))
        for model in (OrchestrationAction, GraphRun, Session, Task, EventLog)
    }

    with pytest.raises(HTTPException) as exc:
        await service.execute_start_graph_action(
            db_session,
            run_id=run.id,
            request={
                "action_type": "start_graph",
                "graph_id": str(graph.id),
                "subject_type": "task",
                "subject_id": str(task.id),
            },
            idempotency_key=f"run:{run.id}:kind:start_graph:unrunnable:{task.id}",
        )

    assert exc.value.status_code == 409
    assert exc.value.detail == {"code": "project_not_runnable", "reason": "workspace_unset"}
    assert {
        model: await db_session.scalar(select(func.count(model.id)))
        for model in counts_before
    } == counts_before
    await db_session.refresh(run)
    await db_session.refresh(gate)
    await db_session.refresh(task)
    assert run.status == "running"
    assert gate.status == "open"
    assert task.status == "in_progress"


@pytest.mark.asyncio
async def test_start_graph_action_rejects_subject_outside_run(db_session, test_project):
    validator = _agent("validator", "validator", ["validation"])
    db_session.add(validator)
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, test_project.id)
    unrelated = Task(
        project_id=test_project.id,
        title="Unrelated task",
        description="Not owned by orchestration.",
        status="in_progress",
        assigned_to=validator.id,
        metadata_={},
    )
    db_session.add(unrelated)
    graph = await _graph(db_session, test_project.id)
    await db_session.flush()

    with pytest.raises(HTTPException) as exc:
        await service.execute_start_graph_action(
            db_session,
            run_id=run.id,
            request={
                "action_type": "start_graph",
                "graph_id": str(graph.id),
                "subject_type": "task",
                "subject_id": str(unrelated.id),
            },
            idempotency_key=f"run:{run.id}:kind:start_graph:unrelated:{unrelated.id}",
        )

    assert exc.value.status_code == 409
    assert await db_session.scalar(select(func.count(GraphRun.id))) == 0
    actions = await _actions(db_session, run.id, "start_graph")
    assert len(actions) == 1
    assert actions[0].status == "failed"


@pytest.mark.asyncio
async def test_concluded_meeting_decision_satisfies_configured_gate(db_session, test_project):
    facilitator = _agent("facilitator", "facilitator", ["meeting_facilitation"])
    reviewer = _agent("reviewer", "reviewer", ["review"])
    db_session.add_all([facilitator, reviewer])
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id, "meeting_decision")
    task = await _make_orchestrated_task(
        db_session,
        test_project.id,
        run.id,
        gate.id,
        reviewer.id,
    )
    action = await service.execute_schedule_meeting_action(
        db_session,
        run_id=run.id,
        request={
            "action_type": "schedule_meeting",
            "topic": "Accept the validation outcome?",
            "participant_agent_ids": [str(facilitator.id), str(reviewer.id)],
            "task_id": str(task.id),
            "gate_id": str(gate.id),
            "organizer_agent_id": str(facilitator.id),
        },
        idempotency_key=f"run:{run.id}:kind:schedule_meeting:gate:{gate.id}",
    )
    meeting = await db_session.get(Meeting, action.target_id)
    agenda_item = (
        await db_session.execute(
            select(MeetingAgendaItem).where(MeetingAgendaItem.meeting_id == meeting.id)
        )
    ).scalar_one()
    meeting.status = "concluded"
    meeting.concluded_at = datetime.now(timezone.utc)
    decision = MeetingDecision(
        meeting_id=meeting.id,
        agenda_item_id=agenda_item.id,
        title="Accept validation outcome",
        question="Does the evidence satisfy the gate?",
        chosen_option="Accept",
        rationale="The independent result resolves the blocker.",
        decided_by="consensus",
    )
    db_session.add(decision)
    await db_session.flush()
    await emit_event_once(
        db_session,
        test_project.id,
        "meeting.concluded",
        {"meeting_id": str(meeting.id)},
        dedup_key=f"phase16-meeting-concluded:{meeting.id}",
    )

    result = await service.tick(db_session, run.id)

    await db_session.refresh(gate)
    evidence = list(
        (
            await db_session.execute(
                select(OrchestrationEvidence).where(OrchestrationEvidence.gate_id == gate.id)
            )
        ).scalars().all()
    )
    assert result["evidence_created"] == 1
    assert result["gates_validated"] == 1
    assert gate.status == "accepted"
    assert evidence[0].source_type == "meeting_decision"
    assert evidence[0].source_id == decision.id
    assert evidence[0].verdict == "accepted"


@pytest.mark.asyncio
async def test_graph_run_step_records_exact_edge_evidence(db_session, test_project):
    validator = _agent("validator", "validator", ["validation"])
    db_session.add(validator)
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id, "graph_run_step")
    task = await _make_orchestrated_task(
        db_session,
        test_project.id,
        run.id,
        gate.id,
        validator.id,
        status="in_progress",
    )
    graph = await _graph(db_session, test_project.id)
    action = await service.execute_start_graph_action(
        db_session,
        run_id=run.id,
        request={
            "action_type": "start_graph",
            "graph_id": str(graph.id),
            "subject_type": "task",
            "subject_id": str(task.id),
        },
        idempotency_key=f"run:{run.id}:kind:start_graph:graph:{graph.id}:task:{task.id}",
    )
    run_instance = await db_session.get(GraphRun, action.target_id)

    before = await service.tick(db_session, run.id)
    assert before["evidence_created"] == 0
    await GraphEngineService().advance_manually(
        db_session,
        run_instance,
        to_node="reviewed",
        reason="Validation completed",
    )
    step = (
        await db_session.execute(
            select(GraphRunStep)
            .where(
                GraphRunStep.graph_run_id == run_instance.id,
                GraphRunStep.to_node == "reviewed",
            )
            .order_by(GraphRunStep.stepped_at.desc())
        )
    ).scalar_one()

    result = await service.tick(db_session, run.id)

    await db_session.refresh(gate)
    evidence = list(
        (
            await db_session.execute(
                select(OrchestrationEvidence).where(OrchestrationEvidence.gate_id == gate.id)
            )
        ).scalars().all()
    )
    assert result["evidence_created"] == 1
    assert gate.status == "accepted"
    assert len(evidence) == 1
    assert evidence[0].source_type == "graph_run_step"
    assert evidence[0].source_id == step.id
    assert evidence[0].evidence_metadata["graph_run_step_id"] == str(step.id)
    assert evidence[0].evidence_metadata["from_node"] == "pending"
    assert evidence[0].evidence_metadata["to_node"] == "reviewed"


async def _start_graph_for_evidence(db_session, project_id, required_source_type):
    validator = _agent("validator", "validator", ["validation"])
    db_session.add(validator)
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, project_id)
    gate = await _make_gate(db_session, run.id, required_source_type)
    task = await _make_orchestrated_task(
        db_session,
        project_id,
        run.id,
        gate.id,
        validator.id,
        status="in_progress",
    )
    graph = await _graph(db_session, project_id)
    action = await service.execute_start_graph_action(
        db_session,
        run_id=run.id,
        request={
            "action_type": "start_graph",
            "graph_id": str(graph.id),
            "subject_type": "task",
            "subject_id": str(task.id),
        },
        idempotency_key=f"run:{run.id}:kind:start_graph:graph:{graph.id}:task:{task.id}",
    )
    run_instance = await db_session.get(GraphRun, action.target_id)
    await service.tick(db_session, run.id)
    return service, run, gate, task, graph, run_instance


@pytest.mark.asyncio
@pytest.mark.parametrize("step_id", [None, "not-a-uuid"], ids=["missing", "malformed"])
async def test_graph_run_step_rejects_missing_or_malformed_step_id(
    db_session,
    test_project,
    step_id,
):
    service, run, gate, _task, graph, run_instance = await _start_graph_for_evidence(
        db_session,
        test_project.id,
        "graph_run_step",
    )
    payload = {
        "graph_run_id": str(run_instance.id),
        "graph_name": graph.name,
        "from_node": "pending",
        "to_node": "reviewed",
    }
    if step_id is not None:
        payload["graph_run_step_id"] = step_id
    await emit_event_once(
        db_session,
        test_project.id,
        "graph.run_advanced",
        payload,
        dedup_key=f"phase16-invalid-step:{run_instance.id}:{step_id}",
    )

    result = await service.tick(db_session, run.id)

    assert result["evidence_created"] == 0
    assert await db_session.scalar(
        select(func.count(OrchestrationEvidence.id)).where(OrchestrationEvidence.gate_id == gate.id)
    ) == 0


@pytest.mark.asyncio
async def test_graph_run_step_rejects_step_owned_by_another_run(db_session, test_project):
    service, run, gate, task, graph, run_instance = await _start_graph_for_evidence(
        db_session,
        test_project.id,
        "graph_run_step",
    )
    foreign_run = GraphRun(
        graph_id=graph.id,
        project_id=test_project.id,
        linked_task_id=task.id,
        current_node="reviewed",
        context={},
    )
    db_session.add(foreign_run)
    await db_session.flush()
    foreign_step = GraphRunStep(
        graph_run_id=foreign_run.id,
        from_node="pending",
        to_node="reviewed",
    )
    db_session.add(foreign_step)
    await db_session.flush()
    await emit_event_once(
        db_session,
        test_project.id,
        "graph.run_advanced",
        {
            "graph_run_id": str(run_instance.id),
            "graph_run_step_id": str(foreign_step.id),
            "graph_name": graph.name,
            "from_node": "pending",
            "to_node": "reviewed",
        },
        dedup_key=f"phase16-foreign-step:{run_instance.id}:{foreign_step.id}",
    )

    result = await service.tick(db_session, run.id)

    assert result["evidence_created"] == 0
    assert await db_session.scalar(
        select(func.count(OrchestrationEvidence.id)).where(OrchestrationEvidence.gate_id == gate.id)
    ) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("include_step_id", [False, True], ids=["legacy-omitted", "owned"])
async def test_terminal_graph_evidence_keeps_run_source(
    db_session,
    test_project,
    include_step_id,
):
    service, run, gate, _task, graph, run_instance = await _start_graph_for_evidence(
        db_session,
        test_project.id,
        "graph_run",
    )
    step = (
        await db_session.execute(
            select(GraphRunStep).where(GraphRunStep.graph_run_id == run_instance.id)
        )
    ).scalar_one()
    run_instance.status = "completed"
    payload = {
        "graph_run_id": str(run_instance.id),
        "graph_name": graph.name,
    }
    if include_step_id:
        payload["graph_run_step_id"] = str(step.id)
    await emit_event_once(
        db_session,
        test_project.id,
        "graph.run_completed",
        payload,
        dedup_key=f"phase16-terminal-source:{run_instance.id}:{include_step_id}",
    )

    result = await service.tick(db_session, run.id)

    evidence = list(
        (
            await db_session.execute(
                select(OrchestrationEvidence).where(OrchestrationEvidence.gate_id == gate.id)
            )
        ).scalars().all()
    )
    assert result["evidence_created"] == 1
    assert len(evidence) == 1
    assert evidence[0].source_type == "graph_run"
    assert evidence[0].source_id == run_instance.id
    assert evidence[0].evidence_metadata["graph_run_step_id"] == (
        str(step.id) if include_step_id else None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event_type", "step_identity"),
    [
        ("graph.run_completed", "foreign"),
        ("graph.run_failed", "foreign"),
        ("graph.run_completed", "malformed"),
    ],
)
async def test_terminal_graph_evidence_rejects_invalid_step_identity(
    db_session,
    test_project,
    event_type,
    step_identity,
):
    service, run, gate, task, graph, run_instance = await _start_graph_for_evidence(
        db_session,
        test_project.id,
        "graph_run",
    )
    if step_identity == "foreign":
        foreign_run = GraphRun(
            graph_id=graph.id,
            project_id=test_project.id,
            linked_task_id=task.id,
            current_node="reviewed",
            context={},
        )
        db_session.add(foreign_run)
        await db_session.flush()
        foreign_step = GraphRunStep(
            graph_run_id=foreign_run.id,
            from_node="pending",
            to_node="reviewed",
        )
        db_session.add(foreign_step)
        await db_session.flush()
        step_id = str(foreign_step.id)
    else:
        step_id = "not-a-uuid"
    await emit_event_once(
        db_session,
        test_project.id,
        event_type,
        {
            "graph_run_id": str(run_instance.id),
            "graph_run_step_id": step_id,
            "graph_name": graph.name,
        },
        dedup_key=f"phase16-invalid-terminal-step:{run_instance.id}:{event_type}:{step_identity}",
    )

    result = await service.tick(db_session, run.id)

    assert result["evidence_created"] == 0
    assert await db_session.scalar(
        select(func.count(OrchestrationEvidence.id)).where(OrchestrationEvidence.gate_id == gate.id)
    ) == 0
