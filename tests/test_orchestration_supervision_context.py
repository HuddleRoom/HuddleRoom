import uuid
from types import SimpleNamespace

import pytest

from huddleroom.models.event_log import EventLog
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGate
from huddleroom.models.orchestration_memory import OrchestrationMemorySection
from huddleroom.models.project import Project
from huddleroom.models.task import Task
from huddleroom.schemas.orchestration import OrchestrationGoalCreate
from huddleroom.services.event_bus import emit_event_once
from huddleroom.services.orchestration_memory_service import OrchestrationMemoryService
from huddleroom.services.orchestration_progress_view import OrchestrationProgressView
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.orchestration_supervision_context import OrchestrationSupervisionContextBuilder


pytestmark = pytest.mark.asyncio


async def _goal_and_run(db_session, project_id):
    return await OrchestrationService().create_goal(
        db_session,
        project_id,
        OrchestrationGoalCreate(objective="Keep durable supervision context"),
        created_by_user_id=None,
    )


async def test_context_uses_durable_rows_not_recent_events(db_session, test_project):
    goal, run = await _goal_and_run(db_session, test_project.id)
    action = OrchestrationAction(
        run_id=run.id,
        idempotency_key="durable-action",
        action_type="create_delegation_task",
        request={"scope": "implementation"},
        dispatch_contract={"deliverable": "a durable result"},
        budget_ledger={},
        status="failed",
        error="runner unavailable",
    )
    db_session.add(action)
    await emit_event_once(
        db_session, test_project.id, "task.status_changed",
        {"task_id": str(uuid.uuid4()), "status": "done", "secret": "transient"},
    )
    await db_session.flush()

    context = await OrchestrationService()._decision_context(db_session, goal, run)

    assert "recent_events" not in context
    assert "event_cursor" not in context["run"]
    assert context["actions"] == [{
        "id": str(action.id), "action_type": "create_delegation_task", "status": "failed",
        "request": {"scope": "implementation"}, "dispatch_contract": {"deliverable": "a durable result"},
        "budget_ledger": {}, "error": "runner unavailable",
    }]


async def test_report_memory_is_unverified_with_complete_provenance(db_session, test_project):
    goal, run = await _goal_and_run(db_session, test_project.id)
    memory = OrchestrationMemoryService()

    section = await memory.upsert_section(
        db_session, test_project.id, goal.id,
        section_key="report_abc", title="Report", body="candidate claim",
        fact_status="unverified",
        provenance={"report_id": "abc", "run_id": str(run.id), "task_id": str(uuid.uuid4()),
                    "session_id": None, "producer_agent_id": None, "event_id": None},
    )

    assert section.fact_status == "unverified"
    assert section.provenance["report_id"] == "abc"
    assert section.provenance["run_id"] == str(run.id)


async def test_context_keeps_all_unverified_memory_beyond_completed_memory_budget(db_session, test_project):
    goal, run = await _goal_and_run(db_session, test_project.id)
    memory = OrchestrationMemoryService()
    for number in range(9):
        await memory.upsert_section(
            db_session, test_project.id, goal.id, section_key=f"report_{number}",
            title="Report", body="candidate", fact_status="unverified",
        )

    context = await OrchestrationService()._decision_context(db_session, goal, run)

    assert len(context["memory"]) == 9


async def test_context_rejects_mismatched_goal_and_run(db_session, test_project):
    goal, run = await _goal_and_run(db_session, test_project.id)
    mismatched_run = SimpleNamespace(id=run.id, goal_id=uuid.uuid4())

    with pytest.raises(ValueError, match="does not belong"):
        await OrchestrationService()._decision_context(db_session, goal, mismatched_run)


async def test_context_excludes_cross_project_action_targets(db_session, test_project):
    goal, run = await _goal_and_run(db_session, test_project.id)
    other_project = Project(name="Other", config={})
    db_session.add(other_project)
    await db_session.flush()
    other_task = Task(project_id=other_project.id, title="Private", status="ready")
    db_session.add(other_task)
    await db_session.flush()
    action = OrchestrationAction(
        run_id=run.id, idempotency_key="other-project-target",
        action_type="create_delegation_task", request={}, target_type="task", target_id=other_task.id,
    )
    db_session.add(action)
    await db_session.flush()

    context = await OrchestrationService()._decision_context(db_session, goal, run)

    assert context["tasks"] == []


async def test_context_excludes_superseded_memory_and_bounds_accepted(db_session, test_project):
    goal, run = await _goal_and_run(db_session, test_project.id)
    memory = OrchestrationMemoryService()
    for number in range(9):
        await memory.upsert_section(
            db_session, test_project.id, goal.id, section_key=f"accepted_{number}",
            title="Accepted", body="settled", fact_status="accepted",
        )
    await memory.upsert_section(
        db_session, test_project.id, goal.id, section_key="superseded",
        title="Old", body="old", fact_status="superseded",
    )

    context = await OrchestrationService()._decision_context(db_session, goal, run)

    assert len(context["memory"]) == 8
    assert {row["fact_status"] for row in context["memory"]} == {"accepted"}


async def test_memory_upsert_defaults_and_only_refreshes_explicit_provenance(db_session, test_project):
    goal, run = await _goal_and_run(db_session, test_project.id)
    memory = OrchestrationMemoryService()
    section = await memory.upsert_section(
        db_session, test_project.id, goal.id, section_key="fact", title="Fact", body="first",
        run_id=run.id, fact_status="accepted", provenance={"source": "first"},
    )
    created_provenance = dict(section.provenance)
    updated = await memory.upsert_section(
        db_session, test_project.id, goal.id, section_key="fact", title="Fact", body="second",
    )

    assert created_provenance["run_id"] == str(run.id)
    assert updated.provenance == created_provenance
    assert updated.fact_status == "accepted"

    refreshed = await memory.upsert_section(
        db_session, test_project.id, goal.id, section_key="fact", title="Fact", body="third",
        provenance={"source": "second"},
    )

    assert refreshed.provenance == {
        "created_by": "orchestrator", "run_id": None, "event_id": None, "source": "second",
    }


async def test_context_marks_legacy_empty_provenance_without_excluding_unverified_memory(db_session, test_project):
    goal, run = await _goal_and_run(db_session, test_project.id)
    legacy = OrchestrationMemorySection(
        project_id=test_project.id, goal_id=goal.id, section_key="legacy", title="Legacy",
        body="unresolved", created_by="legacy", fact_status="unverified", provenance={},
    )
    db_session.add(legacy)
    await db_session.flush()
    updated_at = legacy.updated_at

    context = await OrchestrationService()._decision_context(db_session, goal, run)

    assert context["memory"] == [{
        "id": str(legacy.id), "section_key": "legacy", "title": "Legacy", "summary": None,
        "fact_status": "unverified", "provenance": {
            "legacy": True, "section_id": str(legacy.id), "created_by": "legacy",
        }, "body": "unresolved",
    }]
    assert legacy.provenance == {}
    assert legacy.updated_at == updated_at


async def test_context_build_is_stable_with_open_gates(db_session, test_project):
    goal, run = await _goal_and_run(db_session, test_project.id)
    db_session.add_all([
        OrchestrationGate(run_id=run.id, success_criterion_key="a", gate_type="evidence"),
        OrchestrationGate(run_id=run.id, success_criterion_key="b", gate_type="evidence"),
    ])
    await db_session.flush()

    service = OrchestrationService()
    assert await service._decision_context(db_session, goal, run) == await service._decision_context(db_session, goal, run)


async def test_context_includes_progress_view_for_declared_criteria(db_session, test_project):
    goal, run = await _goal_and_run(db_session, test_project.id)
    goal.success_criteria = [{"key": "alpha", "description": "Alpha works"}]
    db_session.add(OrchestrationGate(run_id=run.id, success_criterion_key="alpha", gate_type="evidence"))
    await db_session.flush()

    context = await OrchestrationSupervisionContextBuilder().build(db_session, goal, run)

    assert [(row["criterion_key"], row["description"], row["state"]) for row in context["progress_view"]] == [
        ("alpha", "Alpha works", "evidence_pending"),
    ]
    assert "progress_view_error" not in context


async def test_context_includes_untracked_follow_ups_and_total(db_session, test_project):
    goal, run = await _goal_and_run(db_session, test_project.id)
    db_session.add_all([
        OrchestrationGate(run_id=run.id, success_criterion_key="a", gate_type="evidence"),
        OrchestrationGate(run_id=run.id, success_criterion_key="b", gate_type="evidence"),
    ])
    await db_session.flush()

    context = await OrchestrationSupervisionContextBuilder().build(db_session, goal, run)

    assert context["untracked_follow_ups_total"] == 2
    assert [row["kind"] for row in context["untracked_follow_ups"]] == ["unverified_gate", "unverified_gate"]


async def test_context_marks_progress_view_error_with_empty_lists(db_session, test_project, monkeypatch):
    goal, run = await _goal_and_run(db_session, test_project.id)
    db_session.add(OrchestrationGate(run_id=run.id, success_criterion_key="a", gate_type="evidence"))
    await db_session.flush()

    async def boom(self, db, goal, run):
        raise RuntimeError("progress read failed")

    monkeypatch.setattr(OrchestrationProgressView, "build", boom)
    context = await OrchestrationSupervisionContextBuilder().build(db_session, goal, run)

    assert context["progress_view_error"] is True
    assert context["progress_view"] == []
    assert context["untracked_follow_ups"] == []
    assert context["untracked_follow_ups_total"] == 0


async def test_context_progress_view_is_stable_across_two_builds(db_session, test_project):
    goal, run = await _goal_and_run(db_session, test_project.id)
    goal.success_criteria = [{"key": "a", "description": "A"}, {"key": "b", "description": "B"}]
    db_session.add_all([
        OrchestrationGate(run_id=run.id, success_criterion_key="a", gate_type="evidence"),
        OrchestrationGate(run_id=run.id, success_criterion_key="b", gate_type="evidence"),
    ])
    await db_session.flush()

    builder = OrchestrationSupervisionContextBuilder()
    first = await builder.build(db_session, goal, run)
    second = await builder.build(db_session, goal, run)

    assert first["progress_view"] == second["progress_view"]
    assert first["untracked_follow_ups"] == second["untracked_follow_ups"]
    assert first["untracked_follow_ups_total"] == second["untracked_follow_ups_total"]
