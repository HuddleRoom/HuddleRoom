import uuid
from datetime import datetime, timezone

import pytest

from huddleroom.models.meeting import Meeting, MeetingActionItem
from huddleroom.models.orchestration import (
    OrchestrationAction,
    OrchestrationDecision,
    OrchestrationGate,
    OrchestrationGoal,
    OrchestrationRun,
    OrchestrationWait,
)
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
from huddleroom.models.task import Task
from huddleroom.services.orchestration_progress_view import (
    ACTIVE_TASK_STATUSES,
    OrchestrationProgressView,
    ProgressSituation,
)
from huddleroom.services.orchestration_wake_when import ORCHESTRATOR_WAIT_OWNER_TYPE

pytestmark = pytest.mark.asyncio


async def _run_with_criteria(db, project, criteria):
    goal = OrchestrationGoal(
        project_id=project.id, objective="Progress view", status="active", goal_type="outcome", success_criteria=criteria,
    )
    db.add(goal)
    await db.flush()
    run = OrchestrationRun(goal_id=goal.id, status="running", phase="authorized")
    db.add(run)
    await db.flush()
    return goal, run


async def _task(db, project, run, status="in_progress", criterion_keys=None):
    task = Task(
        project_id=project.id,
        title=f"Task {uuid.uuid4()}",
        status=status,
        metadata_={"orchestration": {"run_id": str(run.id), "success_criterion_keys": list(criterion_keys or [])}},
    )
    db.add(task)
    await db.flush()
    return task


async def _gate(db, run, key, status="open", required_evidence=None, success_criterion_key=None):
    gate = OrchestrationGate(
        run_id=run.id,
        success_criterion_key=success_criterion_key or key,
        gate_type="work_completed",
        required_evidence=required_evidence or {},
        status=status,
    )
    db.add(gate)
    await db.flush()
    return gate


async def _delegation(db, run, task, inputs):
    action = OrchestrationAction(
        run_id=run.id,
        idempotency_key=f"delegate:{uuid.uuid4()}",
        action_type="create_delegation_task",
        request={"inputs": list(inputs)},
        status="completed",
        target_type="task",
        target_id=task.id,
    )
    db.add(action)
    await db.flush()
    return action


def _entry(situation, key):
    return next(e for e in situation.progress_view if e["criterion_key"] == key)


async def test_no_work_when_nothing_linked(db_session, test_project):
    goal, run = await _run_with_criteria(db_session, test_project, [{"key": "done", "description": "Done."}])
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    entry = _entry(situation, "done")
    assert entry["state"] == "no_work"
    assert entry["task_ids"] == [] and entry["gate_ids"] == []
    assert entry["description"] == "Done."


@pytest.mark.parametrize("status", ACTIVE_TASK_STATUSES)
async def test_in_flight_for_each_active_status(db_session, test_project, status):
    goal, run = await _run_with_criteria(db_session, test_project, [{"key": "done", "description": "Done."}])
    task = await _task(db_session, test_project, run, status=status, criterion_keys=["done"])
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    entry = _entry(situation, "done")
    assert entry["state"] == "in_flight"
    assert entry["task_ids"] == [str(task.id)]


async def test_evidence_pending_when_open_gate_and_no_active_task(db_session, test_project):
    goal, run = await _run_with_criteria(db_session, test_project, [{"key": "done", "description": "Done."}])
    await _task(db_session, test_project, run, status="done", criterion_keys=["done"])
    gate = await _gate(db_session, run, "done", status="open")
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    entry = _entry(situation, "done")
    assert entry["state"] == "evidence_pending"
    assert entry["gate_ids"] == [str(gate.id)]


async def test_evidence_pending_for_failed_gate(db_session, test_project):
    goal, run = await _run_with_criteria(db_session, test_project, [{"key": "done", "description": "Done."}])
    await _gate(db_session, run, "done", status="failed")
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    assert _entry(situation, "done")["state"] == "evidence_pending"


async def test_in_flight_beats_open_gate_when_task_active(db_session, test_project):
    goal, run = await _run_with_criteria(db_session, test_project, [{"key": "done", "description": "Done."}])
    await _task(db_session, test_project, run, status="ready", criterion_keys=["done"])
    await _gate(db_session, run, "done", status="open")
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    assert _entry(situation, "done")["state"] == "in_flight"


async def test_accepted_requires_all_linked_gates_accepted_and_at_least_one(db_session, test_project):
    goal, run = await _run_with_criteria(db_session, test_project, [
        {"key": "done", "description": "Done."},
        {"key": "tested", "description": "Tested."},
        {"key": "empty", "description": "Empty."},
    ])
    await _gate(db_session, run, "done", status="accepted")
    await _gate(db_session, run, "done", status="accepted")
    await _gate(db_session, run, "tested", status="accepted")
    await _gate(db_session, run, "tested", status="open")
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    assert _entry(situation, "done")["state"] == "accepted"
    assert _entry(situation, "tested")["state"] == "evidence_pending"
    assert _entry(situation, "empty")["state"] == "no_work"


async def test_gate_linked_by_required_evidence_keys_and_by_success_criterion_key(db_session, test_project):
    goal, run = await _run_with_criteria(db_session, test_project, [
        {"key": "a", "description": "A."},
        {"key": "b", "description": "B."},
    ])
    by_evidence = await _gate(db_session, run, "shared", status="open", required_evidence={"success_criterion_keys": ["a", "b"]})
    by_key = await _gate(db_session, run, "other", status="open", success_criterion_key="b")
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    assert _entry(situation, "a")["gate_ids"] == [str(by_evidence.id)]
    assert _entry(situation, "b")["gate_ids"] == [str(by_evidence.id), str(by_key.id)]


async def test_gate_with_required_evidence_none_links_by_success_criterion_key(db_session, test_project):
    goal, run = await _run_with_criteria(db_session, test_project, [
        {"key": "a", "description": "A."},
        {"key": "b", "description": "B."},
    ])
    gate = await _gate(db_session, run, "a", status="open", required_evidence=None)
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    assert _entry(situation, "a")["gate_ids"] == [str(gate.id)]
    assert _entry(situation, "b")["gate_ids"] == []


async def test_task_linked_by_delegation_tag_in_inputs(db_session, test_project):
    goal, run = await _run_with_criteria(db_session, test_project, [{"key": "done", "description": "Done."}])
    task = await _task(db_session, test_project, run, status="in_progress")
    await _delegation(db_session, run, task, ["brief", "criterion:done"])
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    entry = _entry(situation, "done")
    assert entry["state"] == "in_flight"
    assert entry["task_ids"] == [str(task.id)]


async def test_done_and_failed_tasks_are_not_in_flight_but_listed_in_task_ids(db_session, test_project):
    goal, run = await _run_with_criteria(db_session, test_project, [{"key": "done", "description": "Done."}])
    finished = await _task(db_session, test_project, run, status="done", criterion_keys=["done"])
    failed = await _task(db_session, test_project, run, status="failed", criterion_keys=["done"])
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    entry = _entry(situation, "done")
    assert entry["state"] == "no_work"
    assert entry["task_ids"] == [str(finished.id), str(failed.id)]


async def test_task_linked_to_two_criteria_appears_in_both(db_session, test_project):
    goal, run = await _run_with_criteria(db_session, test_project, [
        {"key": "a", "description": "A."},
        {"key": "b", "description": "B."},
    ])
    task = await _task(db_session, test_project, run, status="in_progress", criterion_keys=["a", "b"])
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    assert _entry(situation, "a")["task_ids"] == [str(task.id)]
    assert _entry(situation, "b")["task_ids"] == [str(task.id)]
    assert _entry(situation, "a")["state"] == _entry(situation, "b")["state"] == "in_flight"


async def test_blocked_task_is_in_flight_and_stalled(db_session, test_project):
    goal, run = await _run_with_criteria(db_session, test_project, [{"key": "done", "description": "Done."}])
    task = await _task(db_session, test_project, run, status="blocked", criterion_keys=["done"])
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    assert _entry(situation, "done")["state"] == "in_flight"
    assert [f["id"] for f in situation.untracked_follow_ups if f["kind"] == "stalled_task"] == [str(task.id)]


async def test_malformed_criteria_are_skipped(db_session, test_project):
    criteria = [
        "not-a-mapping",
        {"description": "No key."},
        {"key": "", "description": "Empty key."},
        {"key": 7, "description": "Non-string key."},
        {"key": "done", "description": "Done."},
        {"key": "done", "description": "Duplicate."},
    ]
    goal, run = await _run_with_criteria(db_session, test_project, criteria)
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    assert [e["criterion_key"] for e in situation.progress_view] == ["done"]
    assert _entry(situation, "done")["description"] == "Done."


async def test_entries_follow_goal_criteria_order(db_session, test_project):
    goal, run = await _run_with_criteria(db_session, test_project, [
        {"key": "zeta", "description": "Z."},
        {"key": "alpha", "description": "A."},
        {"key": "mid", "description": "M."},
    ])
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    assert [e["criterion_key"] for e in situation.progress_view] == ["zeta", "alpha", "mid"]


async def _meeting(db, project, source_task, title="Standup"):
    meeting = Meeting(project_id=project.id, title=title, meeting_type="standup", source_task_id=source_task.id)
    db.add(meeting)
    await db.flush()
    return meeting


async def _action_item(db, meeting, description="Write the report", **fields):
    item = MeetingActionItem(meeting_id=meeting.id, description=description, **fields)
    db.add(item)
    await db.flush()
    return item


async def _authority_decision(
    db, goal, run, related_action, status="answered", decided_at=None, title="Approve plan", selected_option="approve",
):
    decision = OrchestrationAuthorityDecision(
        goal_id=goal.id,
        run_id=run.id,
        decision_key=f"key-{uuid.uuid4()}",
        title=title,
        status=status,
        authority="human",
        question="Approve?",
        selected_option=selected_option,
        related_action_id=related_action.id if related_action else None,
        decided_at=decided_at,
    )
    db.add(decision)
    await db.flush()
    return decision


async def test_meeting_action_item_included_when_open_untasked_from_run_meeting(db_session, test_project):
    goal, run = await _run_with_criteria(db_session, test_project, [{"key": "done", "description": "Done."}])
    task = await _task(db_session, test_project, run)
    meeting = await _meeting(db_session, test_project, task)
    item = await _action_item(db_session, meeting, description="Write the report")
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    assert situation.untracked_follow_ups_total == 1
    assert situation.untracked_follow_ups == [{
        "kind": "meeting_action_item",
        "id": str(item.id),
        "summary": "Write the report",
        "suggested_actions": ["create_delegation_task"],
    }]
    assert all("created_at" not in follow_up for follow_up in situation.untracked_follow_ups)


async def test_meeting_action_item_tag_matches_uppercase_uuid(db_session, test_project):
    goal, run = await _run_with_criteria(db_session, test_project, [{"key": "done", "description": "Done."}])
    task = await _task(db_session, test_project, run)
    meeting = await _meeting(db_session, test_project, task)
    item = await _action_item(db_session, meeting)
    await _delegation(db_session, run, task, [f"  meeting_action_item:{str(item.id).upper()} "])
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    assert situation.untracked_follow_ups == []


async def test_meeting_action_item_excluded_when_task_linked(db_session, test_project):
    goal, run = await _run_with_criteria(db_session, test_project, [{"key": "done", "description": "Done."}])
    task = await _task(db_session, test_project, run)
    meeting = await _meeting(db_session, test_project, task)
    await _action_item(db_session, meeting, task_id=task.id)
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    assert situation.untracked_follow_ups == []
    assert situation.untracked_follow_ups_total == 0


async def test_meeting_action_item_excluded_when_status_not_open(db_session, test_project):
    goal, run = await _run_with_criteria(db_session, test_project, [{"key": "done", "description": "Done."}])
    task = await _task(db_session, test_project, run)
    meeting = await _meeting(db_session, test_project, task)
    await _action_item(db_session, meeting, status="done")
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    assert situation.untracked_follow_ups == []


async def test_meeting_action_item_excluded_when_graph_already_created(db_session, test_project):
    goal, run = await _run_with_criteria(db_session, test_project, [{"key": "done", "description": "Done."}])
    task = await _task(db_session, test_project, run)
    meeting = await _meeting(db_session, test_project, task)
    await _action_item(db_session, meeting, creates_graph=True, graph_run_id=uuid.uuid4())
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    assert situation.untracked_follow_ups == []


async def test_meeting_action_item_from_other_run_meeting_excluded(db_session, test_project):
    goal, run = await _run_with_criteria(db_session, test_project, [{"key": "done", "description": "Done."}])
    _, other_run = await _run_with_criteria(db_session, test_project, [{"key": "other", "description": "Other."}])
    other_task = await _task(db_session, test_project, other_run)
    other_meeting = await _meeting(db_session, test_project, other_task, title="Other run")
    await _action_item(db_session, other_meeting)
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    assert situation.untracked_follow_ups == []


async def test_meeting_action_item_excluded_when_tagged_by_delegation_action(db_session, test_project):
    goal, run = await _run_with_criteria(db_session, test_project, [{"key": "done", "description": "Done."}])
    task = await _task(db_session, test_project, run)
    meeting = await _meeting(db_session, test_project, task)
    item = await _action_item(db_session, meeting)
    await _delegation(db_session, run, task, [f"meeting_action_item:{item.id}"])
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    assert situation.untracked_follow_ups == []


async def test_answered_decision_included_when_answered_after_latest_decision(db_session, test_project):
    goal, run = await _run_with_criteria(db_session, test_project, [{"key": "done", "description": "Done."}])
    task = await _task(db_session, test_project, run)
    action = await _delegation(db_session, run, task, [])
    db_session.add(OrchestrationDecision(
        run_id=run.id, decision_type="execution", created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    ))
    await db_session.flush()
    decision = await _authority_decision(
        db_session, goal, run, action,
        decided_at=datetime(2026, 1, 2, tzinfo=timezone.utc), title="Approve plan", selected_option="approve",
    )
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    assert situation.untracked_follow_ups == [{
        "kind": "answered_decision",
        "id": str(decision.id),
        "summary": "Approve plan: approve",
        "suggested_actions": [],
    }]


async def test_answered_decision_excluded_when_pending_cancelled_or_expired(db_session, test_project):
    goal, run = await _run_with_criteria(db_session, test_project, [{"key": "done", "description": "Done."}])
    task = await _task(db_session, test_project, run)
    action = await _delegation(db_session, run, task, [])
    decided = datetime(2026, 1, 2, tzinfo=timezone.utc)
    await _authority_decision(db_session, goal, run, action, status="pending", selected_option=None)
    await _authority_decision(db_session, goal, run, action, status="cancelled", decided_at=decided)
    await _authority_decision(db_session, goal, run, action, status="expired", decided_at=decided)
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    assert situation.untracked_follow_ups == []


async def test_answered_decision_excluded_without_related_action(db_session, test_project):
    goal, run = await _run_with_criteria(db_session, test_project, [{"key": "done", "description": "Done."}])
    await _authority_decision(
        db_session, goal, run, None, decided_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    assert situation.untracked_follow_ups == []


async def test_answered_decision_excluded_after_a_later_orchestration_decision(db_session, test_project):
    goal, run = await _run_with_criteria(db_session, test_project, [{"key": "done", "description": "Done."}])
    task = await _task(db_session, test_project, run)
    action = await _delegation(db_session, run, task, [])
    await _authority_decision(
        db_session, goal, run, action, decided_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    db_session.add(OrchestrationDecision(
        run_id=run.id, decision_type="execution", created_at=datetime(2026, 1, 3, tzinfo=timezone.utc),
    ))
    await db_session.flush()
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    assert situation.untracked_follow_ups == []


async def _action(db, run, action_type, request, created_at=None, status="completed"):
    action = OrchestrationAction(
        run_id=run.id,
        idempotency_key=f"{action_type}:{uuid.uuid4()}",
        action_type=action_type,
        request=request,
        status=status,
        created_at=created_at or datetime.now(timezone.utc),
    )
    db.add(action)
    await db.flush()
    return action


async def test_failed_and_blocked_tasks_are_stalled(db_session, test_project):
    goal, run = await _run_with_criteria(db_session, test_project, [{"key": "done", "description": "Done."}])
    failed = await _task(db_session, test_project, run, status="failed")
    blocked = await _task(db_session, test_project, run, status="blocked")
    await _task(db_session, test_project, run, status="done")
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    stalled = [f for f in situation.untracked_follow_ups if f["kind"] == "stalled_task"]
    assert [f["id"] for f in stalled] == [str(failed.id), str(blocked.id)]
    assert stalled[0]["summary"] == f"{failed.title} is failed"
    assert stalled[0]["suggested_actions"] == ["retry_task", "reassign_task"]


async def test_stalled_task_suggested_actions_depend_on_status(db_session, test_project):
    goal, run = await _run_with_criteria(db_session, test_project, [{"key": "done", "description": "Done."}])
    await _task(db_session, test_project, run, status="failed")
    await _task(db_session, test_project, run, status="blocked")
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    actions = {f["summary"].rsplit(" is ", 1)[1]: f["suggested_actions"] for f in situation.untracked_follow_ups}
    assert actions == {"failed": ["retry_task", "reassign_task"], "blocked": []}


@pytest.mark.parametrize("action_type", ["retry_task", "reassign_task", "request_split"])
async def test_stalled_task_tracked_by_later_retry_or_reassign_or_split_action(db_session, test_project, action_type):
    goal, run = await _run_with_criteria(db_session, test_project, [{"key": "done", "description": "Done."}])
    task = await _task(db_session, test_project, run, status="failed")
    task.updated_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    await db_session.flush()
    await _action(
        db_session, run, action_type, {"task_id": str(task.id)},
        created_at=datetime(2026, 1, 2, tzinfo=timezone.utc), status="reserved",
    )
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    assert [f for f in situation.untracked_follow_ups if f["kind"] == "stalled_task"] == []


async def test_stalled_task_flagged_again_when_task_updated_after_action(db_session, test_project):
    goal, run = await _run_with_criteria(db_session, test_project, [{"key": "done", "description": "Done."}])
    task = await _task(db_session, test_project, run, status="failed")
    await _action(
        db_session, run, "retry_task", {"task_id": str(task.id)},
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    task.updated_at = datetime(2026, 1, 2, tzinfo=timezone.utc)
    await db_session.flush()
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    assert [f["id"] for f in situation.untracked_follow_ups if f["kind"] == "stalled_task"] == [str(task.id)]


async def test_done_task_is_not_stalled(db_session, test_project):
    goal, run = await _run_with_criteria(db_session, test_project, [{"key": "done", "description": "Done."}])
    await _task(db_session, test_project, run, status="done")
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    assert situation.untracked_follow_ups == []


async def test_open_gate_without_verification_request_is_unverified(db_session, test_project):
    goal, run = await _run_with_criteria(db_session, test_project, [{"key": "done", "description": "Done."}])
    gate = await _gate(db_session, run, "done", status="open")
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    assert situation.untracked_follow_ups == [{
        "kind": "unverified_gate",
        "id": str(gate.id),
        "summary": "work_completed gate for done has no verification request",
        "suggested_actions": ["request_verification"],
    }]


async def test_gate_excluded_when_request_verification_names_it(db_session, test_project):
    goal, run = await _run_with_criteria(db_session, test_project, [{"key": "done", "description": "Done."}])
    gate = await _gate(db_session, run, "done", status="open")
    await _action(db_session, run, "request_verification", {"gate_id": str(gate.id)}, status="reserved")
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    assert [f for f in situation.untracked_follow_ups if f["kind"] == "unverified_gate"] == []


async def test_gate_excluded_when_active_producer_task_points_at_it(db_session, test_project):
    goal, run = await _run_with_criteria(db_session, test_project, [{"key": "done", "description": "Done."}])
    gate = await _gate(db_session, run, "done", status="open")
    db_session.add(Task(
        project_id=test_project.id,
        title="Producer",
        status="in_progress",
        metadata_={"orchestration": {"run_id": str(run.id), "plan_item_gate_id": str(gate.id)}},
    ))
    await db_session.flush()
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    assert [f for f in situation.untracked_follow_ups if f["kind"] == "unverified_gate"] == []


async def test_gate_excluded_when_producer_task_failed(db_session, test_project):
    goal, run = await _run_with_criteria(db_session, test_project, [{"key": "done", "description": "Done."}])
    gate = await _gate(db_session, run, "done", status="open")
    db_session.add(Task(
        project_id=test_project.id,
        title="Failed producer",
        status="failed",
        metadata_={"orchestration": {"run_id": str(run.id), "plan_item_gate_id": str(gate.id)}},
    ))
    await db_session.flush()
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    assert [f for f in situation.untracked_follow_ups if f["kind"] == "unverified_gate"] == []


async def test_accepted_gate_is_not_unverified(db_session, test_project):
    goal, run = await _run_with_criteria(db_session, test_project, [{"key": "done", "description": "Done."}])
    await _gate(db_session, run, "done", status="accepted")
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    assert situation.untracked_follow_ups == []


async def test_follow_ups_follow_source_then_created_order(db_session, test_project):
    goal, run = await _run_with_criteria(db_session, test_project, [{"key": "done", "description": "Done."}])
    gate = await _gate(db_session, run, "done", status="open")
    failed = await _task(db_session, test_project, run, status="failed")
    task = await _task(db_session, test_project, run)
    action = await _delegation(db_session, run, task, [])
    decision = await _authority_decision(
        db_session, goal, run, action, decided_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    meeting = await _meeting(db_session, test_project, task)
    late = await _action_item(db_session, meeting, description="Late", created_at=datetime(2026, 1, 5, tzinfo=timezone.utc))
    early = await _action_item(db_session, meeting, description="Early", created_at=datetime(2026, 1, 3, tzinfo=timezone.utc))
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    assert [(f["kind"], f["id"]) for f in situation.untracked_follow_ups] == [
        ("meeting_action_item", str(early.id)),
        ("meeting_action_item", str(late.id)),
        ("answered_decision", str(decision.id)),
        ("stalled_task", str(failed.id)),
        ("unverified_gate", str(gate.id)),
    ]


async def test_cap_is_ten_and_total_counts_all(db_session, test_project):
    goal, run = await _run_with_criteria(db_session, test_project, [{"key": "done", "description": "Done."}])
    task = await _task(db_session, test_project, run)
    meeting = await _meeting(db_session, test_project, task)
    for i in range(12):
        await _action_item(db_session, meeting, description=f"Item {i}", created_at=datetime(2026, 1, 1 + i, tzinfo=timezone.utc))
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    assert situation.untracked_follow_ups_total == 12
    assert len(situation.untracked_follow_ups) == 10
    assert [f["summary"] for f in situation.untracked_follow_ups] == [f"Item {i}" for i in range(10)]


async def _orchestrator_wait(db, run, item_id, owner_type=ORCHESTRATOR_WAIT_OWNER_TYPE, status="open"):
    wait = OrchestrationWait(
        run_id=run.id,
        wait_key=f"wait-{uuid.uuid4()}",
        owner={"type": owner_type},
        awaited_event={"event_type": "decision.answered", "matcher": {"item_id": str(item_id)}},
        due_recheck_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
        fallback={},
        status=status,
    )
    db.add(wait)
    await db.flush()
    return wait


async def test_item_excluded_when_orchestrator_wait_matcher_equals_its_id(db_session, test_project):
    goal, run = await _run_with_criteria(db_session, test_project, [{"key": "done", "description": "Done."}])
    task = await _task(db_session, test_project, run)
    meeting = await _meeting(db_session, test_project, task)
    item = await _action_item(db_session, meeting)
    await _orchestrator_wait(db_session, run, item.id)
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    assert situation.untracked_follow_ups == []
    assert situation.untracked_follow_ups_total == 0


async def test_stalled_task_excluded_when_orchestrator_wait_matcher_equals_its_id(db_session, test_project):
    goal, run = await _run_with_criteria(db_session, test_project, [{"key": "done", "description": "Done."}])
    task = await _task(db_session, test_project, run, status="failed")
    await _orchestrator_wait(db_session, run, task.id)
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    assert situation.untracked_follow_ups == []


async def test_unverified_gate_excluded_when_orchestrator_wait_matcher_equals_its_id(db_session, test_project):
    goal, run = await _run_with_criteria(db_session, test_project, [{"key": "done", "description": "Done."}])
    gate = await _gate(db_session, run, "done", status="open")
    await _orchestrator_wait(db_session, run, gate.id)
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    assert situation.untracked_follow_ups == []


async def test_system_owned_wait_does_not_track_items(db_session, test_project):
    goal, run = await _run_with_criteria(db_session, test_project, [{"key": "done", "description": "Done."}])
    task = await _task(db_session, test_project, run)
    meeting = await _meeting(db_session, test_project, task)
    item = await _action_item(db_session, meeting)
    await _orchestrator_wait(db_session, run, item.id, owner_type="system")
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    assert [f["id"] for f in situation.untracked_follow_ups] == [str(item.id)]


async def test_cleared_orchestrator_wait_does_not_track(db_session, test_project):
    goal, run = await _run_with_criteria(db_session, test_project, [{"key": "done", "description": "Done."}])
    task = await _task(db_session, test_project, run)
    meeting = await _meeting(db_session, test_project, task)
    item = await _action_item(db_session, meeting)
    await _orchestrator_wait(db_session, run, item.id, status="cleared")
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    assert [f["id"] for f in situation.untracked_follow_ups] == [str(item.id)]


async def test_build_safe_returns_error_flag_and_empty_lists(db_session, test_project, monkeypatch):
    goal, run = await _run_with_criteria(db_session, test_project, [{"key": "done", "description": "Done."}])

    async def boom(self, db, goal, run):
        raise RuntimeError("boom")

    monkeypatch.setattr(OrchestrationProgressView, "build", boom)
    situation = await OrchestrationProgressView().build_safe(db_session, goal, run)
    assert situation == ProgressSituation(error=True)
    assert situation.progress_view == [] and situation.untracked_follow_ups == []
    assert situation.untracked_follow_ups_total == 0


async def test_as_context_shape_with_and_without_error():
    assert ProgressSituation().as_context() == {
        "progress_view": [], "untracked_follow_ups": [], "untracked_follow_ups_total": 0,
    }
    assert ProgressSituation(error=True).as_context() == {
        "progress_view": [], "untracked_follow_ups": [], "untracked_follow_ups_total": 0,
        "progress_view_error": True,
    }


async def test_two_builds_without_changes_are_equal(db_session, test_project):
    goal, run = await _run_with_criteria(db_session, test_project, [{"key": "done", "description": "Done."}])
    task = await _task(db_session, test_project, run, status="failed", criterion_keys=["done"])
    meeting = await _meeting(db_session, test_project, task)
    await _action_item(db_session, meeting)
    await _gate(db_session, run, "done", status="open")
    view = OrchestrationProgressView()
    assert await view.build(db_session, goal, run) == await view.build(db_session, goal, run)
