from dataclasses import FrozenInstanceError
import uuid

import pytest

from huddleroom.models.orchestration import OrchestrationGate, OrchestrationGoal, OrchestrationRun
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
from huddleroom.models.orchestration_steering import OrchestrationSteeringRequest
from huddleroom.models.graph import Graph, GraphRun
from huddleroom.models.meeting import Meeting
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.services.orchestration_wake_source import (
    LiveWorkRefV1,
    OrchestrationWakeSourceService,
    SourceRef,
    WakeSourceVerdict,
)


pytestmark = pytest.mark.asyncio


async def _run(db, project):
    goal = OrchestrationGoal(project_id=project.id, objective="Wake source", status="active")
    db.add(goal)
    await db.flush()
    run = OrchestrationRun(goal_id=goal.id, status="running", phase="authorized")
    db.add(run)
    await db.flush()
    return goal, run


async def _task(db, project, run, *, status="blocked", depends_on=None):
    task = Task(
        project_id=project.id,
        title=f"Task {uuid.uuid4()}",
        status=status,
        depends_on=[str(task_id) for task_id in depends_on or []],
        metadata_={"orchestration": {"run_id": str(run.id)}},
    )
    db.add(task)
    await db.flush()
    return task


async def _gated_task(db, project, run, *, status="ready", depends_on=None):
    gate = OrchestrationGate(run_id=run.id, success_criterion_key=str(uuid.uuid4()), gate_type="work_completed")
    db.add(gate)
    await db.flush()
    task = await _task(db, project, run, status=status, depends_on=depends_on)
    task.metadata_ = {"orchestration": {"run_id": str(run.id), "gate_id": str(gate.id)}}
    await db.flush()
    return task


def _source(task):
    return SourceRef(kind="task", id=task.id)


def _ref(kind, producer, source):
    return LiveWorkRefV1(
        kind=kind,
        id=producer.id,
        blocked_source_kind=source.kind,
        blocked_source_id=source.id,
    )


def _event(event_type, **matcher):
    return {"events": [{"event_type": event_type, "matcher": {key: str(value) for key, value in matcher.items()}}],
            "expected_result": "The relevant producer advances."}


def _timer():
    return {"recheck_after_seconds": 60, "expected_result": "The relevant producer advances."}


async def _evaluate(db, run, source, wake_when, refs):
    return await OrchestrationWakeSourceService().evaluate(
        db, run, blocked_source=source, wake_when=wake_when, live_work_refs=refs,
    )


async def test_wake_source_public_contract_is_frozen():
    source = SourceRef("task", uuid.uuid4())
    ref = LiveWorkRefV1("task", uuid.uuid4(), source.kind, source.id)
    verdict = WakeSourceVerdict("live", "matched_live_producer", ref, {"event_type": "task.status_changed"})

    with pytest.raises(FrozenInstanceError):
        source.kind = "goal"
    with pytest.raises(FrozenInstanceError):
        ref.kind = "session"
    with pytest.raises(FrozenInstanceError):
        verdict.verdict = "eventless"


async def test_only_relevant_independently_runnable_source_is_live(db_session, test_project):
    _, run = await _run(db_session, test_project)
    producer = await _gated_task(db_session, test_project, run, status="ready")
    blocked = await _task(db_session, test_project, run, depends_on=[producer.id])
    unrelated = await _gated_task(db_session, test_project, run, status="in_progress")
    source = _source(blocked)
    wake_when = _event("task.status_changed", task_id=producer.id)

    relevant = await _evaluate(db_session, run, source, wake_when, [_ref("task", producer, source)])
    unrelated_verdict = await _evaluate(db_session, run, source, wake_when, [_ref("task", unrelated, source)])

    assert relevant.verdict == "live"
    assert relevant.producer == _ref("task", producer, source)
    assert relevant.matched_event == wake_when["events"][0]
    assert unrelated_verdict.verdict == "eventless"
    assert unrelated_verdict.producer is None


async def test_blocked_self_cyclic_and_terminal_task_producers_are_eventless(db_session, test_project):
    _, run = await _run(db_session, test_project)
    upstream = await _gated_task(db_session, test_project, run, status="ready")
    blocked_producer = await _gated_task(db_session, test_project, run, status="blocked", depends_on=[upstream.id])
    blocked = await _task(db_session, test_project, run, depends_on=[blocked_producer.id])
    cycle_source = await _task(db_session, test_project, run)
    cyclic = await _gated_task(db_session, test_project, run, status="blocked", depends_on=[cycle_source.id])
    cycle_source.depends_on = [str(cyclic.id)]
    terminal = await _gated_task(db_session, test_project, run, status="done")
    source = _source(blocked)

    blocked_verdict = await _evaluate(
        db_session, run, source, _event("task.status_changed", task_id=blocked_producer.id),
        [_ref("task", blocked_producer, source)],
    )
    cyclic_verdict = await _evaluate(
        db_session, run, _source(cycle_source), _event("task.status_changed", task_id=cyclic.id),
        [_ref("task", cyclic, _source(cycle_source))],
    )
    self_reference = await _evaluate(
        db_session, run, source, _event("task.status_changed", task_id=blocked.id),
        [_ref("task", blocked, source)],
    )
    terminal_verdict = await _evaluate(
        db_session, run, source, _event("task.status_changed", task_id=terminal.id),
        [_ref("task", terminal, source)],
    )

    assert blocked_verdict.reason_code == "producer_not_independently_runnable"
    assert cyclic_verdict.reason_code == "producer_not_independently_runnable"
    assert self_reference.reason_code == "producer_not_independently_runnable"
    assert terminal_verdict.verdict == "eventless"


async def test_session_and_timer_require_the_exact_blocked_source(db_session, test_project, test_agent):
    _, run = await _run(db_session, test_project)
    blocked = await _task(db_session, test_project, run)
    unrelated = await _task(db_session, test_project, run)
    session = Session(project_id=test_project.id, task_id=blocked.id, agent_id=test_agent.id, adapter_type="api", status="running")
    unrelated_session = Session(
        project_id=test_project.id, task_id=unrelated.id, agent_id=test_agent.id, adapter_type="api", status="running",
    )
    db_session.add_all([session, unrelated_session])
    await db_session.flush()
    source = _source(blocked)

    event_verdict = await _evaluate(
        db_session, run, source, _event("session.completed", session_id=session.id), [_ref("session", session, source)],
    )
    task_event_verdict = await _evaluate(
        db_session, run, source, _event("task.status_changed", task_id=blocked.id), [_ref("session", session, source)],
    )
    wrong_task_event = await _evaluate(
        db_session, run, source, _event("task.status_changed", task_id=unrelated.id), [_ref("session", session, source)],
    )
    timer_verdict = await _evaluate(db_session, run, source, _timer(), [_ref("session", session, source)])
    wrong_source = await _evaluate(
        db_session, run, source, _timer(), [_ref("session", session, _source(unrelated))],
    )
    claimed_owner = await _evaluate(
        db_session, run, source, _event("session.completed", session_id=unrelated_session.id),
        [_ref("session", unrelated_session, source)],
    )
    no_ref = await _evaluate(db_session, run, source, _timer(), [])

    assert event_verdict.verdict == "live"
    assert event_verdict.matched_event["matcher"] == {"session_id": str(session.id)}
    assert task_event_verdict.verdict == "live"
    assert task_event_verdict.matched_event["matcher"] == {"task_id": str(blocked.id)}
    assert wrong_task_event.verdict == "eventless"
    assert timer_verdict.verdict == "live"
    assert timer_verdict.matched_event is None
    assert wrong_source.verdict == claimed_owner.verdict == no_ref.verdict == "eventless"


async def test_missing_graph_and_meeting_refs_are_eventless_and_active_owned_rows_are_live(db_session, test_project):
    _, run = await _run(db_session, test_project)
    blocked = await _task(db_session, test_project, run)
    source = _source(blocked)
    graph = Graph(project_id=test_project.id, name=f"Wake graph {uuid.uuid4()}", version="1", definition={})
    db_session.add(graph)
    await db_session.flush()
    graph_run = GraphRun(
        graph_id=graph.id, project_id=test_project.id, linked_task_id=blocked.id, current_node="start", status="active",
    )
    meeting = Meeting(
        project_id=test_project.id, title="Wake meeting", meeting_type="review", source_task_id=blocked.id, status="active",
    )
    db_session.add_all([graph_run, meeting])
    await db_session.flush()

    graph_verdict = await _evaluate(
        db_session, run, source, _event("graph.run_completed", graph_run_id=graph_run.id),
        [_ref("graph_run", graph_run, source)],
    )
    meeting_verdict = await _evaluate(
        db_session, run, source, _event("meeting.concluded", meeting_id=meeting.id),
        [_ref("meeting", meeting, source)],
    )
    missing = await _evaluate(
        db_session, run, source, _timer(),
        [LiveWorkRefV1("task", uuid.uuid4(), source.kind, source.id)],
    )

    assert graph_verdict.verdict == meeting_verdict.verdict == "live"
    assert graph_verdict.matched_event["matcher"] == {"graph_run_id": str(graph_run.id)}
    assert meeting_verdict.matched_event["matcher"] == {"meeting_id": str(meeting.id)}
    assert missing.verdict == "eventless"


async def test_pending_steering_and_human_decision_are_live_only_with_matching_wakes(
    db_session, test_project, test_user,
):
    goal, run = await _run(db_session, test_project)
    blocked = await _task(db_session, test_project, run)
    source = _source(blocked)
    steering = OrchestrationSteeringRequest(
        id=uuid.uuid4(), goal_id=goal.id, actor_id=test_user.id, client_request_id=uuid.uuid4(), sequence=1,
        submitted_run_id=run.id, directive="Work on the blocked task", target_type="task", target_id=str(blocked.id),
        scope="item", lifetime="selected_item", impact_summary="Unblocks the task", status="pending",
        reason_code="submitted", contract_version="test", plan_version="test",
    )
    decision = OrchestrationAuthorityDecision(
        goal_id=goal.id, run_id=run.id, decision_key="wake-source-human", title="Choose", authority="human",
        question="Choose a direction.", options=[{"key": "continue"}], status="pending",
    )
    db_session.add_all([steering, decision])
    await db_session.flush()

    steering_verdict = await _evaluate(
        db_session, run, source, _timer(), [_ref("steering_request", steering, source)],
    )
    decision_verdict = await _evaluate(
        db_session, run, source, _event("authority.decision_resolved", decision_id=decision.id),
        [_ref("authority_decision", decision, source)],
    )
    mismatched_decision = await _evaluate(
        db_session, run, source, _event("task.status_changed", task_id=blocked.id),
        [_ref("authority_decision", decision, source)],
    )

    assert steering_verdict.verdict == "live"
    assert steering_verdict.matched_event is None
    assert decision_verdict.verdict == "live"
    assert decision_verdict.matched_event["event_type"] == "authority.decision_resolved"
    assert mismatched_decision.verdict == "eventless"
