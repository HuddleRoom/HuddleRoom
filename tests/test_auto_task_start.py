"""Tasks created for an agent automatically start immediately (in_progress + auto Session)."""
import pytest
from sqlalchemy import select

from huddleroom.models.graph import Graph, GraphRun
from huddleroom.models.meeting import MeetingActionItem
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.routers import agent_self_service as svc


async def _auto_sessions(db, task_id):
    return list((await db.scalars(select(Session).where(Session.task_id == task_id, Session.origin == "auto"))).all())


@pytest.mark.asyncio
async def test_report_subtask_assigned_starts_unassigned_stays_backlog(db_session, test_project, test_agent):
    sess = Session(project_id=test_project.id, agent_id=test_agent.id, adapter_type="api", status="completed")
    db_session.add(sess)
    await db_session.flush()
    await svc.post_agent_report(
        svc.AgentReport(
            session_id=sess.id,
            subtasks=[
                svc.SubtaskReport(title="assigned-st", assigned_to=test_agent.id),
                svc.SubtaskReport(title="unassigned-st"),
            ],
        ),
        test_agent,
        db_session,
    )
    a = await db_session.scalar(select(Task).where(Task.title == "assigned-st"))
    u = await db_session.scalar(select(Task).where(Task.title == "unassigned-st"))
    assert a.status == "in_progress" and len(await _auto_sessions(db_session, a.id)) == 1
    assert u.status == "backlog" and await _auto_sessions(db_session, u.id) == []


@pytest.mark.asyncio
async def test_meeting_action_items_assigned_starts_unassigned_stays_backlog(db_session, test_project, test_agent):
    from huddleroom.services.meeting_outcome import MeetingOutcomeService
    from huddleroom.services.meeting_service import MeetingService

    meeting = await MeetingService().create_meeting(
        db=db_session, project_id=test_project.id, title="Auto start", meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        agenda_items=[{"order": 1, "title": "Item", "max_rounds": 1}],
    )
    items = [
        MeetingActionItem(meeting_id=meeting.id, description="assigned-ai", assignee_agent_id=test_agent.id, priority=50),
        MeetingActionItem(meeting_id=meeting.id, description="unassigned-ai", priority=50),
    ]
    db_session.add_all(items)
    await db_session.flush()
    a, u = await MeetingOutcomeService().create_tasks_from_action_items(db_session, meeting, items)
    assert a.status == "in_progress" and len(await _auto_sessions(db_session, a.id)) == 1
    assert u.status == "backlog" and await _auto_sessions(db_session, u.id) == []


@pytest.mark.asyncio
async def test_graph_assign_task_starts_immediately(db_session, test_project, test_agent):
    from huddleroom.services.action_executor import ActionExecutor

    graph = Graph(project_id=None, name="t-auto-assign", version="1.0", definition={}, triggers=[], is_active=True)
    db_session.add(graph)
    await db_session.flush()
    run = GraphRun(
        graph_id=graph.id, project_id=test_project.id, current_node="n", status="active",
        actor_assignments={"dev": {"kind": "agent", "id": str(test_agent.id)}}, context={},
    )
    db_session.add(run)
    await db_session.flush()
    res = await ActionExecutor().execute(
        db_session, {"action_type": "assign_task", "to_actor": "dev", "task_title": "graph-t"}, run
    )
    task = await db_session.scalar(select(Task).where(Task.title == "graph-t"))
    assert task.assigned_to == test_agent.id
    assert task.status == "in_progress" and len(await _auto_sessions(db_session, task.id)) == 1


@pytest.mark.asyncio
async def test_meeting_action_item_for_inactive_agent_stays_backlog(db_session, test_project, test_agent):
    from huddleroom.services.meeting_outcome import MeetingOutcomeService
    from huddleroom.services.meeting_service import MeetingService

    test_agent.is_active = False
    await db_session.flush()
    meeting = await MeetingService().create_meeting(
        db=db_session, project_id=test_project.id, title="Inactive", meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        agenda_items=[{"order": 1, "title": "Item", "max_rounds": 1}],
    )
    item = MeetingActionItem(meeting_id=meeting.id, description="inactive-ai", assignee_agent_id=test_agent.id, priority=50)
    db_session.add(item)
    await db_session.flush()
    (task,) = await MeetingOutcomeService().create_tasks_from_action_items(db_session, meeting, [item])
    assert task.status == "backlog" and await _auto_sessions(db_session, task.id) == []


@pytest.mark.asyncio
async def test_meeting_action_item_session_failure_does_not_abort(db_session, test_project, test_agent, monkeypatch):
    from fastapi import HTTPException
    from huddleroom.services.meeting_outcome import MeetingOutcomeService
    from huddleroom.services.meeting_service import MeetingService
    from huddleroom.services.session_service import SessionService

    async def _refuse(self, db, data):
        raise HTTPException(status_code=409, detail="refused")

    monkeypatch.setattr(SessionService, "create", _refuse)
    meeting = await MeetingService().create_meeting(
        db=db_session, project_id=test_project.id, title="Refused", meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        agenda_items=[{"order": 1, "title": "Item", "max_rounds": 1}],
    )
    item = MeetingActionItem(meeting_id=meeting.id, description="refused-ai", assignee_agent_id=test_agent.id, priority=50)
    db_session.add(item)
    await db_session.flush()
    (task,) = await MeetingOutcomeService().create_tasks_from_action_items(db_session, meeting, [item])
    assert task.status == "backlog" and await _auto_sessions(db_session, task.id) == []
