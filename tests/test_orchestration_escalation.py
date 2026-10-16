import uuid

import pytest
from sqlalchemy import func, select

from huddleroom.models.agent import Agent
from huddleroom.models.orchestration import OrchestrationAction
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
from huddleroom.models.task import Task
from huddleroom.schemas.orchestration import OrchestrationGoalCreate
from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
from huddleroom.services.orchestration_service import OrchestrationService

pytestmark = pytest.mark.asyncio

CATEGORIES = [
    "owner_decision", "permission_required", "capability_unavailable", "budget_exhausted",
    "result_correction_exhausted", "revision_exhausted", "review_input_missing",
]


async def _make_run(db, project_id, budget=None):
    from tests.conftest import complete_baseline_processes

    service = OrchestrationService()
    goal, run = await service.create_goal(
        db,
        project_id=project_id,
        data=OrchestrationGoalCreate(
            objective="Ship work",
            success_criteria=[{"key": "done", "description": "Work is done."}],
            budget=budget if budget is not None else {"caps": {"max_tokens": 20001}},
        ),
        created_by_user_id=None,
    )
    run.baseline_authorized = True
    await complete_baseline_processes(db, goal, run)
    goal.status, run.status, run.phase = "active", "running", "authorized"
    await db.flush()
    return service, goal, run


async def _asks(db, run):
    return list((await db.execute(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == "ask_human"
    ))).scalars())


@pytest.mark.parametrize("category", CATEGORIES)
async def test_escalation_names_category_subject_and_attempts(db_session, test_project, category):
    service, goal, run = await _make_run(db_session, test_project.id)
    subject = f"task:{uuid.uuid4()}:thing"
    action = await service._escalate_owner(
        db_session, goal, run, category, subject, ["first try failed", "second try failed"], "Please decide.",
    )
    question = action.request["question"]
    assert category in question and subject in question
    assert "1. first try failed" in question and "2. second try failed" in question
    assert question != "Capability needed"
    blocker = next(b for b in run.active_blockers if b["kind"] == "owner_escalation")
    assert blocker["category"] == category and blocker["decision_id"] == str(action.target_id)
    assert blocker["subject"] == subject and blocker["history"] == ["first try failed", "second try failed"]
    assert run.status == "running" and goal.status == "active"


async def test_escalation_is_idempotent_per_subject(db_session, test_project):
    service, goal, run = await _make_run(db_session, test_project.id)
    first = await service._escalate_owner(db_session, goal, run, "revision_exhausted", "gate:a", ["x"], "Q")
    again = await service._escalate_owner(db_session, goal, run, "revision_exhausted", "gate:a", ["x"], "Q")
    other = await service._escalate_owner(db_session, goal, run, "revision_exhausted", "gate:b", ["y"], "Q")
    assert again.id == first.id and other.id != first.id
    assert len(await _asks(db_session, run)) == 2
    assert len([b for b in run.active_blockers if b["kind"] == "owner_escalation"]) == 2


async def test_unknown_category_rejected(db_session, test_project):
    service, goal, run = await _make_run(db_session, test_project.id)
    with pytest.raises(ValueError):
        await service._escalate_owner(db_session, goal, run, "bogus", "s", [], "Q")


async def test_unrelated_ready_delegation_still_dispatches_after_escalation(db_session, test_project):
    agent = Agent(name=f"dev-{uuid.uuid4()}", role="developer", provider="openai", model="gpt-4o-mini",
                  adapter_type="api", capabilities=["implementation"], config={}, is_active=True)
    db_session.add(agent)
    await db_session.flush()
    service, goal, run = await _make_run(db_session, test_project.id)
    await service._escalate_owner(
        db_session, goal, run, "capability_unavailable",
        f"task:{uuid.uuid4()}:capability:browser", ["no agent"], "Need browser.",
    )
    action = await service.execute_create_delegation_task_action(
        db_session, run.id,
        {
            "action_type": "create_delegation_task", "agent_id": str(agent.id), "work_function": "implementation",
            "scope": "Independent work.", "inputs": ["README.md"], "deliverable": "A change.",
            "success_evidence": ["pytest output"], "budget": {"max_tokens": 20000},
            "report_schema": {"changed_files": "list[str]"},
        },
        f"run:{run.id}:kind:create_delegation_task:independent",
    )
    assert action.status == "completed"
    assert await db_session.get(Task, action.id) is not None


async def _answer(db, action, user):
    decision = await db.get(OrchestrationAuthorityDecision, action.target_id)
    await OrchestrationAuthorityDecisionService().answer_runtime_question(
        db, decision, "acknowledge", actor_user_id=user.id, contract_version=decision.contract_version,
    )


async def test_owner_answer_clears_blocker_and_recall_does_not_resurrect_it(db_session, test_project, test_user):
    service, goal, run = await _make_run(db_session, test_project.id)
    action = await service._escalate_owner(db_session, goal, run, "budget_exhausted", "run:budget", ["capped"], "Raise?")
    assert [b["category"] for b in run.active_blockers] == ["budget_exhausted"]
    await _answer(db_session, action, test_user)
    await db_session.refresh(run)
    assert not run.active_blockers
    again = await service._escalate_owner(db_session, goal, run, "budget_exhausted", "run:budget", ["capped"], "Raise?")
    assert again.id == action.id and not run.active_blockers


async def test_cancelled_owner_ask_drops_only_its_own_blocker(db_session, test_project):
    service, goal, run = await _make_run(db_session, test_project.id)
    cancelled = await service._escalate_owner(db_session, goal, run, "revision_exhausted", "gate:a", ["x"], "Q")
    await service._escalate_owner(db_session, goal, run, "revision_exhausted", "gate:b", ["y"], "Q")
    decision = await db_session.get(OrchestrationAuthorityDecision, cancelled.target_id)
    await OrchestrationAuthorityDecisionService().cancel_decision(db_session, decision, reason="superseded")
    assert [b["scope"] for b in run.active_blockers if b["kind"] == "owner_escalation"] == ["gate:b"]
    again = await service._escalate_owner(db_session, goal, run, "revision_exhausted", "gate:a", ["x"], "Q")
    assert again.id != cancelled.id  # SPR 215: a cancelled ask opens a new cycle and raises its blocker again
    assert sorted(b["scope"] for b in run.active_blockers if b["kind"] == "owner_escalation") == ["gate:a", "gate:b"]


async def test_long_subject_key_is_hashed_and_history_capped(db_session, test_project):
    service, goal, run = await _make_run(db_session, test_project.id)
    subject = "task:" + "x" * 400
    action = await service._escalate_owner(
        db_session, goal, run, "owner_decision", subject, [f"try {i}" for i in range(25)], "Q",
    )
    assert len(action.idempotency_key) <= 255 and subject in action.request["question"]
    assert run.active_blockers[0]["history"] == [f"try {i}" for i in range(15, 25)]


async def test_tick_with_escalation_blocker_still_dispatches_unrelated_ready_task(db_session, test_project):
    agent = Agent(name=f"dev-{uuid.uuid4()}", role="developer", provider="openai", model="gpt-4o-mini",
                  adapter_type="api", capabilities=["implementation"], config={}, is_active=True)
    db_session.add(agent)
    await db_session.flush()
    service, goal, run = await _make_run(db_session, test_project.id, budget={})
    db_session.add(OrchestrationAction(  # effective Start authorization gates activation
        run_id=run.id, idempotency_key=f"run:{run.id}:kind:authorize_execution",
        action_type="authorize_execution", status="completed"))
    task = Task(project_id=test_project.id, title="ready", status="backlog", assigned_to=agent.id,
                metadata_={"orchestration": {"run_id": str(run.id)}})
    db_session.add(task)
    await db_session.flush()
    await service._escalate_owner(db_session, goal, run, "capability_unavailable", "task:other:capability:browser", ["none"], "Q")
    await service.tick(db_session, run.id)
    assert (await db_session.get(Task, task.id)).status == "in_progress"
    assert any(b["kind"] == "owner_escalation" for b in run.active_blockers)
