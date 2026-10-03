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
from huddleroom.models.protocol import Protocol, ProtocolInstance, ProtocolTimeout, ProtocolTransition
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.schemas.orchestration import OrchestrationGoalCreate
from huddleroom.services.event_bus import emit_event_once
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.protocol_engine import ProtocolEngineService


@pytest_asyncio.fixture(autouse=True)
async def _runnable_protocol_project(db_session, test_project, tmp_path):
    workspace = tmp_path / "protocol-workspace"
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
        description="Work linked to a meeting protocol gate.",
        status=status,
        assigned_to=agent_id,
        started_at=datetime.now(timezone.utc),
        metadata_={
            "orchestration": {
                "goal_id": "meeting-protocol-goal",
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


async def _protocol(db_session, project_id) -> Protocol:
    protocol = Protocol(
        project_id=project_id,
        name=f"meeting-protocol-review-{uuid.uuid4()}",
        version="1.0",
        definition={
            "initial_state": "pending",
            "states": {
                "pending": {
                    "timeout": {"duration": "1h", "action": "escalate"},
                    "transitions": [],
                },
                "reviewed": {"transitions": []},
            },
            "terminal_states": {"success": [], "failure": []},
        },
        triggers=[],
        is_active=True,
    )
    db_session.add(protocol)
    await db_session.flush()
    return protocol


@pytest.mark.asyncio
async def test_start_protocol_action_uses_engine_and_replays_once(db_session, test_project):
    validator = _agent("validator", "validator", ["validation"])
    db_session.add(validator)
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id, "protocol_transition")
    task = await _make_orchestrated_task(
        db_session,
        test_project.id,
        run.id,
        gate.id,
        validator.id,
        status="in_progress",
    )
    protocol = await _protocol(db_session, test_project.id)
    request = {
        "action_type": "start_protocol",
        "protocol_id": str(protocol.id),
        "subject_type": "task",
        "subject_id": str(task.id),
    }
    key = f"run:{run.id}:kind:start_protocol:protocol:{protocol.id}:task:{task.id}"

    first = await service.execute_start_protocol_action(
        db_session,
        run_id=run.id,
        request=request,
        idempotency_key=key,
    )
    second = await service.execute_start_protocol_action(
        db_session,
        run_id=run.id,
        request=request,
        idempotency_key=key,
    )

    assert first.id == second.id
    assert first.status == "completed"
    assert first.target_type == "protocol_instance"
    instance = await db_session.get(ProtocolInstance, first.target_id)
    assert instance is not None
    assert instance.protocol_id == protocol.id
    assert instance.linked_task_id == task.id
    assert instance.current_state == "pending"
    assert instance.triggering_event_id == first.id
    transitions = list(
        (
            await db_session.execute(
                select(ProtocolTransition).where(ProtocolTransition.protocol_instance_id == instance.id)
            )
        ).scalars().all()
    )
    assert len(transitions) == 1
    assert transitions[0].from_state == ""
    assert transitions[0].to_state == "pending"
    assert await db_session.scalar(
        select(func.count(ProtocolTimeout.id)).where(ProtocolTimeout.protocol_instance_id == instance.id)
    ) == 1
    assert await db_session.scalar(select(func.count(ProtocolInstance.id))) == 1
    assert len(await _events(db_session, test_project.id, "protocol.instance_started")) == 1
    orchestration_events = await _events(
        db_session,
        test_project.id,
        "orchestration.protocol_started",
    )
    assert len(orchestration_events) == 1
    assert orchestration_events[0].payload["gate_id"] == str(gate.id)
    assert await db_session.scalar(
        select(func.count(OrchestrationEvidence.id)).where(OrchestrationEvidence.run_id == run.id)
    ) == 0


@pytest.mark.asyncio
async def test_start_protocol_action_rejects_unrunnable_project_before_reservation(db_session, test_project):
    """Moving the guard after reservation would persist a failed action for a rejected execution."""
    validator = _agent("validator", "validator", ["validation"])
    db_session.add(validator)
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, test_project.id)
    test_project.workspace_path = None
    await db_session.flush()
    gate = await _make_gate(db_session, run.id, "protocol_transition")
    task = await _make_orchestrated_task(
        db_session,
        test_project.id,
        run.id,
        gate.id,
        validator.id,
        status="in_progress",
    )
    protocol = await _protocol(db_session, test_project.id)
    counts_before = {
        model: await db_session.scalar(select(func.count(model.id)))
        for model in (OrchestrationAction, ProtocolInstance, Session, Task, EventLog)
    }

    with pytest.raises(HTTPException) as exc:
        await service.execute_start_protocol_action(
            db_session,
            run_id=run.id,
            request={
                "action_type": "start_protocol",
                "protocol_id": str(protocol.id),
                "subject_type": "task",
                "subject_id": str(task.id),
            },
            idempotency_key=f"run:{run.id}:kind:start_protocol:unrunnable:{task.id}",
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
async def test_start_protocol_action_rejects_subject_outside_run(db_session, test_project):
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
    protocol = await _protocol(db_session, test_project.id)
    await db_session.flush()

    with pytest.raises(HTTPException) as exc:
        await service.execute_start_protocol_action(
            db_session,
            run_id=run.id,
            request={
                "action_type": "start_protocol",
                "protocol_id": str(protocol.id),
                "subject_type": "task",
                "subject_id": str(unrelated.id),
            },
            idempotency_key=f"run:{run.id}:kind:start_protocol:unrelated:{unrelated.id}",
        )

    assert exc.value.status_code == 409
    assert await db_session.scalar(select(func.count(ProtocolInstance.id))) == 0
    actions = await _actions(db_session, run.id, "start_protocol")
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
async def test_protocol_transition_records_exact_transition_evidence(db_session, test_project):
    validator = _agent("validator", "validator", ["validation"])
    db_session.add(validator)
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id, "protocol_transition")
    task = await _make_orchestrated_task(
        db_session,
        test_project.id,
        run.id,
        gate.id,
        validator.id,
        status="in_progress",
    )
    protocol = await _protocol(db_session, test_project.id)
    action = await service.execute_start_protocol_action(
        db_session,
        run_id=run.id,
        request={
            "action_type": "start_protocol",
            "protocol_id": str(protocol.id),
            "subject_type": "task",
            "subject_id": str(task.id),
        },
        idempotency_key=f"run:{run.id}:kind:start_protocol:protocol:{protocol.id}:task:{task.id}",
    )
    instance = await db_session.get(ProtocolInstance, action.target_id)

    before = await service.tick(db_session, run.id)
    assert before["evidence_created"] == 0
    await ProtocolEngineService().advance_manually(
        db_session,
        instance,
        "reviewed",
        reason="Validation completed",
    )
    transition = (
        await db_session.execute(
            select(ProtocolTransition)
            .where(
                ProtocolTransition.protocol_instance_id == instance.id,
                ProtocolTransition.to_state == "reviewed",
            )
            .order_by(ProtocolTransition.transitioned_at.desc())
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
    assert evidence[0].source_type == "protocol_transition"
    assert evidence[0].source_id == transition.id
    assert evidence[0].evidence_metadata["protocol_transition_id"] == str(transition.id)
    assert evidence[0].evidence_metadata["from_state"] == "pending"
    assert evidence[0].evidence_metadata["to_state"] == "reviewed"


async def _start_protocol_for_evidence(db_session, project_id, required_source_type):
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
    protocol = await _protocol(db_session, project_id)
    action = await service.execute_start_protocol_action(
        db_session,
        run_id=run.id,
        request={
            "action_type": "start_protocol",
            "protocol_id": str(protocol.id),
            "subject_type": "task",
            "subject_id": str(task.id),
        },
        idempotency_key=f"run:{run.id}:kind:start_protocol:protocol:{protocol.id}:task:{task.id}",
    )
    instance = await db_session.get(ProtocolInstance, action.target_id)
    await service.tick(db_session, run.id)
    return service, run, gate, task, protocol, instance


@pytest.mark.asyncio
@pytest.mark.parametrize("transition_id", [None, "not-a-uuid"], ids=["missing", "malformed"])
async def test_protocol_state_transition_rejects_missing_or_malformed_transition_id(
    db_session,
    test_project,
    transition_id,
):
    service, run, gate, _task, protocol, instance = await _start_protocol_for_evidence(
        db_session,
        test_project.id,
        "protocol_transition",
    )
    payload = {
        "protocol_instance_id": str(instance.id),
        "protocol_name": protocol.name,
        "from_state": "pending",
        "to_state": "reviewed",
    }
    if transition_id is not None:
        payload["protocol_transition_id"] = transition_id
    await emit_event_once(
        db_session,
        test_project.id,
        "protocol.state_transitioned",
        payload,
        dedup_key=f"phase16-invalid-transition:{instance.id}:{transition_id}",
    )

    result = await service.tick(db_session, run.id)

    assert result["evidence_created"] == 0
    assert await db_session.scalar(
        select(func.count(OrchestrationEvidence.id)).where(OrchestrationEvidence.gate_id == gate.id)
    ) == 0


@pytest.mark.asyncio
async def test_protocol_state_transition_rejects_transition_owned_by_another_instance(db_session, test_project):
    service, run, gate, task, protocol, instance = await _start_protocol_for_evidence(
        db_session,
        test_project.id,
        "protocol_transition",
    )
    foreign_instance = ProtocolInstance(
        protocol_id=protocol.id,
        project_id=test_project.id,
        linked_task_id=task.id,
        current_state="reviewed",
        context={},
    )
    db_session.add(foreign_instance)
    await db_session.flush()
    foreign_transition = ProtocolTransition(
        protocol_instance_id=foreign_instance.id,
        from_state="pending",
        to_state="reviewed",
    )
    db_session.add(foreign_transition)
    await db_session.flush()
    await emit_event_once(
        db_session,
        test_project.id,
        "protocol.state_transitioned",
        {
            "protocol_instance_id": str(instance.id),
            "protocol_transition_id": str(foreign_transition.id),
            "protocol_name": protocol.name,
            "from_state": "pending",
            "to_state": "reviewed",
        },
        dedup_key=f"phase16-foreign-transition:{instance.id}:{foreign_transition.id}",
    )

    result = await service.tick(db_session, run.id)

    assert result["evidence_created"] == 0
    assert await db_session.scalar(
        select(func.count(OrchestrationEvidence.id)).where(OrchestrationEvidence.gate_id == gate.id)
    ) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("include_transition_id", [False, True], ids=["legacy-omitted", "owned"])
async def test_terminal_protocol_evidence_keeps_instance_source(
    db_session,
    test_project,
    include_transition_id,
):
    service, run, gate, _task, protocol, instance = await _start_protocol_for_evidence(
        db_session,
        test_project.id,
        "protocol_instance",
    )
    transition = (
        await db_session.execute(
            select(ProtocolTransition).where(ProtocolTransition.protocol_instance_id == instance.id)
        )
    ).scalar_one()
    instance.status = "completed"
    payload = {
        "protocol_instance_id": str(instance.id),
        "protocol_name": protocol.name,
    }
    if include_transition_id:
        payload["protocol_transition_id"] = str(transition.id)
    await emit_event_once(
        db_session,
        test_project.id,
        "protocol.completed",
        payload,
        dedup_key=f"phase16-terminal-source:{instance.id}:{include_transition_id}",
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
    assert evidence[0].source_type == "protocol_instance"
    assert evidence[0].source_id == instance.id
    assert evidence[0].evidence_metadata["protocol_transition_id"] == (
        str(transition.id) if include_transition_id else None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event_type", "transition_identity"),
    [
        ("protocol.completed", "foreign"),
        ("protocol.failed", "foreign"),
        ("protocol.completed", "malformed"),
    ],
)
async def test_terminal_protocol_evidence_rejects_invalid_transition_identity(
    db_session,
    test_project,
    event_type,
    transition_identity,
):
    service, run, gate, task, protocol, instance = await _start_protocol_for_evidence(
        db_session,
        test_project.id,
        "protocol_instance",
    )
    if transition_identity == "foreign":
        foreign_instance = ProtocolInstance(
            protocol_id=protocol.id,
            project_id=test_project.id,
            linked_task_id=task.id,
            current_state="reviewed",
            context={},
        )
        db_session.add(foreign_instance)
        await db_session.flush()
        foreign_transition = ProtocolTransition(
            protocol_instance_id=foreign_instance.id,
            from_state="pending",
            to_state="reviewed",
        )
        db_session.add(foreign_transition)
        await db_session.flush()
        transition_id = str(foreign_transition.id)
    else:
        transition_id = "not-a-uuid"
    await emit_event_once(
        db_session,
        test_project.id,
        event_type,
        {
            "protocol_instance_id": str(instance.id),
            "protocol_transition_id": transition_id,
            "protocol_name": protocol.name,
        },
        dedup_key=f"phase16-invalid-terminal-transition:{instance.id}:{event_type}:{transition_identity}",
    )

    result = await service.tick(db_session, run.id)

    assert result["evidence_created"] == 0
    assert await db_session.scalar(
        select(func.count(OrchestrationEvidence.id)).where(OrchestrationEvidence.gate_id == gate.id)
    ) == 0
