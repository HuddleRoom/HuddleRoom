import uuid
from datetime import timedelta
import json

from huddleroom.models.base import _utcnow
from huddleroom.models.orchestration import (
    OrchestrationAction,
    OrchestrationEvidence,
    OrchestrationGate,
    OrchestrationGoal,
    OrchestrationRun,
)
from huddleroom.models.orchestration_process import OrchestrationProcessRun, OrchestrationWarning
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.services import orchestration_effectiveness_review
from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
from huddleroom.services.orchestration_effectiveness_analyzer import (
    EffectivenessAnalysis,
    EffectivenessAnalyzer,
)
from huddleroom.services.orchestration_memory_service import OrchestrationMemoryService
from huddleroom.services.orchestration_process_service import OrchestrationProcessService
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.orchestration_warning_service import OrchestrationWarningService
import pytest


class _EffectivenessAnalyzerStub:
    """Stub EffectivenessAnalyzer returning a canned disposition.

    Counts invocations and raises if called more than `max_calls` times, so a
    wiring regression where a short-circuit path wrongly invokes the analyzer
    fails loudly instead of silently returning the canned value.
    """

    def __init__(self, disposition, *, max_calls=1, rationale="stub rationale"):
        self.disposition = disposition
        self.rationale = rationale
        self.max_calls = max_calls
        self.calls = 0

    @staticmethod
    def build_request(payload, project=None):
        return EffectivenessAnalyzer.build_request(payload, project=project)

    async def review_request(self, request, *, project_id=None):
        self.calls += 1
        if self.calls > self.max_calls:
            raise AssertionError(
                f"EffectivenessAnalyzer stub called {self.calls} times, expected at most {self.max_calls}"
            )
        return EffectivenessAnalysis(self.disposition, (), self.rationale)

    async def review(self, payload, project=None, *, project_id=None):
        return await self.review_request(
            self.build_request(payload, project=project), project_id=project_id
        )


class _EffectivenessAnalyzerFailThenSucceedStub:
    """Stub whose review_request raises until flipped to succeed.

    Drives the analyzer-failure/recovery branches of _advance_running and
    retry_failed: same request-building delegation as _EffectivenessAnalyzerStub,
    but starts in a failing state (should_fail=True) that a test can flip off
    to exercise recovery.
    """

    def __init__(self, *, error=None, disposition="pause", rationale="stub rationale"):
        self.error = error or RuntimeError("provider unavailable")
        self.disposition = disposition
        self.rationale = rationale
        self.should_fail = True
        self.calls = 0

    @staticmethod
    def build_request(payload, project=None):
        return EffectivenessAnalyzer.build_request(payload, project=project)

    async def review_request(self, request, *, project_id=None):
        self.calls += 1
        if self.should_fail:
            raise self.error
        return EffectivenessAnalysis(self.disposition, (), self.rationale)


async def _goal_run(db_session, test_project):
    goal = OrchestrationGoal(
        project_id=test_project.id,
        objective="Review effectiveness",
        success_criteria=[],
        constraints={},
        budget={},
    )
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id)
    db_session.add(run)
    await db_session.flush()
    return goal, run


@pytest.mark.asyncio
async def test_recovery_trigger_emits_every_qualifying_gate_in_stable_order(
    db_session, test_project
):
    goal, run = await _goal_run(db_session, test_project)
    first_task = Task(
        project_id=test_project.id,
        title="Implement first",
        status="in_progress",
    )
    second_task = Task(
        project_id=test_project.id,
        title="Implement second",
        status="in_progress",
    )
    first_gate_id = uuid.uuid4()
    second_gate_id = uuid.uuid4()
    db_session.add_all((first_task, second_task))
    await db_session.flush()
    run.plan_state = {
        "expanded_items": [
            {"task_id": str(first_task.id), "gate_id": str(first_gate_id)},
            {"task_id": str(second_task.id), "gate_id": str(second_gate_id)},
        ]
    }
    db_session.add_all(
        OrchestrationAction(
            run_id=run.id,
            idempotency_key=f"recovery-{task.id}-{offset}",
            action_type="retry_task" if offset % 2 else "reassign_task",
            request={"task_id": str(task.id)},
        )
        for task, count in ((first_task, 3), (second_task, 2))
        for offset in range(count)
    )
    await db_session.flush()

    triggers = await orchestration_effectiveness_review.detect_triggers(db_session, goal, run)

    assert [(trigger.name, trigger.token) for trigger in triggers] == [
        ("repeated_recovery", token)
        for token in sorted(
            (
                f"gate:{first_gate_id}:3",
                f"gate:{second_gate_id}:2",
            )
        )
    ]


def _inactivity_tokens(triggers):
    return [trigger.token for trigger in triggers if trigger.name == "inactivity"]


NON_PROGRESS_ACTION_TYPES = (
    "noop",
    "record_warning",
    "acknowledge_warning",
    "resolve_warning",
    "pause_run",
    "ask_human",
    "suggest_agent",
    "decision_continuation",
)


@pytest.mark.asyncio
async def test_repeated_noops_do_not_reset_inactivity(db_session, test_project):
    goal, run = await _goal_run(db_session, test_project)
    run_started = _utcnow() - timedelta(hours=25)
    run.started_at = run_started
    db_session.add_all(
        OrchestrationAction(
            run_id=run.id,
            idempotency_key=f"noop-{offset}",
            action_type="noop",
            status="completed",
            request={},
            created_at=_utcnow() - timedelta(hours=1),
        )
        for offset in range(3)
    )
    await db_session.flush()

    triggers = await orchestration_effectiveness_review.detect_triggers(db_session, goal, run)

    assert _inactivity_tokens(triggers) == [run_started.isoformat()]


@pytest.mark.asyncio
@pytest.mark.parametrize("action_type", NON_PROGRESS_ACTION_TYPES)
async def test_non_progress_action_types_do_not_reset_inactivity(db_session, test_project, action_type):
    goal, run = await _goal_run(db_session, test_project)
    run_started = _utcnow() - timedelta(hours=25)
    run.started_at = run_started
    db_session.add(
        OrchestrationAction(
            run_id=run.id,
            idempotency_key="non-progress",
            action_type=action_type,
            status="completed",
            request={},
            created_at=_utcnow() - timedelta(hours=1),
        )
    )
    await db_session.flush()

    triggers = await orchestration_effectiveness_review.detect_triggers(db_session, goal, run)

    assert _inactivity_tokens(triggers) == [run_started.isoformat()]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["reserved", "failed"])
async def test_uncompleted_progress_action_does_not_reset_inactivity(db_session, test_project, status):
    goal, run = await _goal_run(db_session, test_project)
    run_started = _utcnow() - timedelta(hours=25)
    run.started_at = run_started
    db_session.add(
        OrchestrationAction(
            run_id=run.id,
            idempotency_key="uncompleted",
            action_type="retry_task",
            status=status,
            request={},
            created_at=_utcnow() - timedelta(hours=1),
        )
    )
    await db_session.flush()

    triggers = await orchestration_effectiveness_review.detect_triggers(db_session, goal, run)

    assert _inactivity_tokens(triggers) == [run_started.isoformat()]


@pytest.mark.asyncio
async def test_completed_progress_action_resets_inactivity(db_session, test_project):
    goal, run = await _goal_run(db_session, test_project)
    run.started_at = _utcnow() - timedelta(hours=25)
    db_session.add(
        OrchestrationAction(
            run_id=run.id,
            idempotency_key="progress",
            action_type="retry_task",
            status="completed",
            request={},
            created_at=_utcnow() - timedelta(hours=1),
        )
    )
    await db_session.flush()

    triggers = await orchestration_effectiveness_review.detect_triggers(db_session, goal, run)

    assert _inactivity_tokens(triggers) == []


@pytest.mark.asyncio
async def test_task_completed_at_resets_inactivity(db_session, test_project):
    goal, run = await _goal_run(db_session, test_project)
    run.started_at = _utcnow() - timedelta(hours=25)
    task = Task(
        project_id=test_project.id,
        title="Completed recently",
        status="done",
        completed_at=_utcnow() - timedelta(hours=1),
    )
    db_session.add(task)
    await db_session.flush()
    run.plan_state = {"expanded_items": [{"task_id": str(task.id), "gate_id": "gate-one"}]}
    await db_session.flush()

    triggers = await orchestration_effectiveness_review.detect_triggers(db_session, goal, run)

    assert _inactivity_tokens(triggers) == []


@pytest.mark.asyncio
async def test_task_started_at_from_run_metadata_resets_inactivity(db_session, test_project):
    goal, run = await _goal_run(db_session, test_project)
    run.started_at = _utcnow() - timedelta(hours=25)
    db_session.add(
        Task(
            project_id=test_project.id,
            title="Started via metadata",
            status="in_progress",
            started_at=_utcnow() - timedelta(hours=1),
            metadata_={"orchestration": {"run_id": str(run.id)}},
        )
    )
    await db_session.flush()

    triggers = await orchestration_effectiveness_review.detect_triggers(db_session, goal, run)

    assert _inactivity_tokens(triggers) == []


@pytest.mark.asyncio
async def test_bare_task_updated_at_bump_does_not_reset_inactivity(db_session, test_project):
    goal, run = await _goal_run(db_session, test_project)
    run_started = _utcnow() - timedelta(hours=25)
    run.started_at = run_started
    task = Task(
        project_id=test_project.id,
        title="Touched recently",
        status="in_progress",
        updated_at=_utcnow(),
    )
    db_session.add(task)
    await db_session.flush()
    run.plan_state = {"expanded_items": [{"task_id": str(task.id), "gate_id": "gate-one"}]}
    await db_session.flush()

    triggers = await orchestration_effectiveness_review.detect_triggers(db_session, goal, run)

    assert _inactivity_tokens(triggers) == [run_started.isoformat()]


@pytest.mark.asyncio
@pytest.mark.parametrize("timestamp_field", ["accepted_at", "failed_at"])
async def test_gate_decision_resets_inactivity(db_session, test_project, timestamp_field):
    goal, run = await _goal_run(db_session, test_project)
    run.started_at = _utcnow() - timedelta(hours=25)
    gate_status = "accepted" if timestamp_field == "accepted_at" else "failed"
    db_session.add_all(
        (
            OrchestrationGate(
                run_id=run.id,
                success_criterion_key="decided",
                gate_type="implementation",
                status=gate_status,
                **{timestamp_field: _utcnow() - timedelta(hours=1)},
            ),
            OrchestrationGate(
                run_id=run.id,
                success_criterion_key="still-open",
                gate_type="implementation",
                status="open",
            ),
        )
    )
    await db_session.flush()

    triggers = await orchestration_effectiveness_review.detect_triggers(db_session, goal, run)

    assert _inactivity_tokens(triggers) == []


@pytest.mark.asyncio
async def test_evidence_updated_at_resets_inactivity(db_session, test_project):
    goal, run = await _goal_run(db_session, test_project)
    run.started_at = _utcnow() - timedelta(hours=25)
    gate = OrchestrationGate(
        run_id=run.id,
        success_criterion_key="evidenced",
        gate_type="implementation",
        status="open",
    )
    db_session.add(gate)
    await db_session.flush()
    db_session.add(
        OrchestrationEvidence(
            run_id=run.id,
            gate_id=gate.id,
            source_type="test_run",
            created_at=_utcnow() - timedelta(hours=25),
            updated_at=_utcnow() - timedelta(hours=1),
        )
    )
    await db_session.flush()

    triggers = await orchestration_effectiveness_review.detect_triggers(db_session, goal, run)

    assert _inactivity_tokens(triggers) == []


@pytest.mark.asyncio
async def test_manager_check_rejects_inconsistent_missing_and_inactive_selected_principals(
    db_session, test_project, test_agent, test_user
):
    goal, run = await _goal_run(db_session, test_project)
    goal.objective = "Defined"
    goal.success_criteria = [{"description": "Observable"}]

    not_selected = (await orchestration_effectiveness_review.collect_checks(db_session, goal, run))[1]
    assert not_selected.passed
    assert not_selected.detail == "Manager authority has not been selected yet."

    goal.manager_agent_id = test_agent.id
    inconsistent_preselection = (await orchestration_effectiveness_review.collect_checks(db_session, goal, run))[1]
    assert not inconsistent_preselection.passed
    assert f"agent_id={test_agent.id}" in inconsistent_preselection.detail

    goal.authority_model = "agent_manager"
    goal.manager_agent_id = test_agent.id
    assert (await orchestration_effectiveness_review.collect_checks(db_session, goal, run))[1].passed

    goal.manager_agent_id = None
    goal.manager_user_id = test_user.id
    inconsistent = (await orchestration_effectiveness_review.collect_checks(db_session, goal, run))[1]
    assert not inconsistent.passed
    assert f"user_id={test_user.id}" in inconsistent.detail

    missing_id = uuid.uuid4()
    missing_goal = OrchestrationGoal(
        id=uuid.uuid4(),
        project_id=test_project.id,
        objective="Defined",
        success_criteria=[{"description": "Observable"}],
        constraints={},
        budget={},
        authority_model="agent_manager",
        manager_agent_id=missing_id,
    )
    missing_run = OrchestrationRun(goal_id=missing_goal.id, plan_state={})
    missing = (await orchestration_effectiveness_review.collect_checks(db_session, missing_goal, missing_run))[1]
    assert not missing.passed
    assert f"agent_id={missing_id}" in missing.detail

    test_agent.is_active = False
    goal.manager_user_id = None
    goal.manager_agent_id = test_agent.id
    inactive_agent = (await orchestration_effectiveness_review.collect_checks(db_session, goal, run))[1]
    assert not inactive_agent.passed

    test_user.is_active = False
    goal.authority_model = "human_manager"
    goal.manager_agent_id = None
    goal.manager_user_id = test_user.id
    inactive_user = (await orchestration_effectiveness_review.collect_checks(db_session, goal, run))[1]
    assert not inactive_user.passed


@pytest.mark.asyncio
@pytest.mark.parametrize("waiting", [False, True])
async def test_hierarchy_check_normalizes_outputs_and_validates_only_stored_role_principals(
    db_session, test_project, waiting
):
    goal, run = await _goal_run(db_session, test_project)
    goal.authority_model = "no_manager"
    missing_id = uuid.uuid4()
    proposal = {
        "role_to_agent": {"implementation": str(missing_id)},
        "weak_fits": [],
    }
    outputs = {"proposal": proposal, "fingerprint": "do-not-rebuild"} if waiting else proposal
    db_session.add(
        OrchestrationProcessRun(
            goal_id=goal.id,
            run_id=run.id,
            process_type="team_hierarchy",
            status="waiting_decision" if waiting else "completed",
            trigger_reason="test",
            input_snapshot={},
            outputs=outputs,
        )
    )
    await db_session.flush()

    check = (await orchestration_effectiveness_review.collect_checks(db_session, goal, run))[2]

    assert not check.passed
    assert check.detail == f"Invalid stored team hierarchy principals: implementation=agent:{missing_id}."


@pytest.mark.asyncio
async def test_live_weak_fit_check_matches_expanded_work_function_assignment_and_live_status(
    db_session, test_project, test_agent
):
    goal, run = await _goal_run(db_session, test_project)
    goal.authority_model = "no_manager"
    live = Task(
        project_id=test_project.id,
        title="Live weak fit",
        status="blocked",
        assigned_to=test_agent.id,
    )
    done = Task(
        project_id=test_project.id,
        title="Done weak fit",
        status="done",
        assigned_to=test_agent.id,
    )
    db_session.add_all((live, done))
    await db_session.flush()
    run.plan_state = {
        "expanded_items": [
            {"task_id": str(live.id), "work_function": "implementation"},
            {"task_id": str(done.id), "work_function": "implementation"},
        ]
    }
    db_session.add(
        OrchestrationProcessRun(
            goal_id=goal.id,
            run_id=run.id,
            process_type="team_hierarchy",
            status="completed",
            trigger_reason="test",
            input_snapshot={},
            outputs={
                "role_to_agent": {"implementation": str(test_agent.id)},
                "weak_fits": [
                    {"work_function": "implementation", "agent_id": str(test_agent.id)},
                    {"work_function": "review", "agent_id": str(test_agent.id)},
                ],
            },
        )
    )
    await db_session.flush()

    check = (await orchestration_effectiveness_review.collect_checks(db_session, goal, run))[3]

    assert not check.passed
    assert check.detail == (
        f"Live weak-fit assignments: task:{live.id}:implementation:agent:{test_agent.id}."
    )


@pytest.mark.asyncio
async def test_material_warning_requires_active_acknowledgement_and_current_failed_gate(
    db_session, test_project
):
    goal, run = await _goal_run(db_session, test_project)
    goal.authority_model = "no_manager"
    failed_gate = OrchestrationGate(
        run_id=run.id,
        success_criterion_key="failed",
        gate_type="evidence",
        required_evidence={},
        status="failed",
    )
    accepted_gate = OrchestrationGate(
        run_id=run.id,
        success_criterion_key="accepted",
        gate_type="evidence",
        required_evidence={},
        status="accepted",
    )
    db_session.add_all((failed_gate, accepted_gate))
    await db_session.flush()
    qualifying = OrchestrationWarning(
        goal_id=goal.id,
        run_id=run.id,
        warning_type="accepted_risk",
        severity="warning",
        message="risk",
        related_gate_id=failed_gate.id,
        acknowledged_by="human:test",
        acknowledged_at=_utcnow(),
    )
    db_session.add_all(
        (
            qualifying,
            OrchestrationWarning(
                goal_id=goal.id,
                run_id=run.id,
                warning_type="not_acknowledged",
                severity="warning",
                message="risk",
                related_gate_id=failed_gate.id,
            ),
            OrchestrationWarning(
                goal_id=goal.id,
                run_id=run.id,
                warning_type="resolved",
                severity="warning",
                message="risk",
                related_gate_id=failed_gate.id,
                acknowledged_by="human:test",
                acknowledged_at=_utcnow(),
                active=False,
            ),
            OrchestrationWarning(
                goal_id=goal.id,
                run_id=run.id,
                warning_type="gate_recovered",
                severity="warning",
                message="risk",
                related_gate_id=accepted_gate.id,
                acknowledged_by="human:test",
                acknowledged_at=_utcnow(),
            ),
        )
    )
    await db_session.flush()

    check = (await orchestration_effectiveness_review.collect_checks(db_session, goal, run))[4]

    assert not check.passed
    assert check.detail == (
        f"Materialized acknowledged warnings: warning:{qualifying.id}:gate:{failed_gate.id}."
    )


@pytest.mark.asyncio
async def test_no_current_inactivity_starts_parks_and_routes_to_active_agent_manager(
    db_session, test_project, test_agent
):
    goal, run = await _goal_run(db_session, test_project)
    goal.success_criteria = [{"description": "Observable result"}]
    goal.authority_model = "agent_manager"
    goal.manager_agent_id = test_agent.id
    run.started_at = _utcnow() - timedelta(hours=25)
    stub = _EffectivenessAnalyzerStub("pause")
    process = orchestration_effectiveness_review.EffectivenessReviewProcess(stub)

    summary = await process.advance(db_session, goal, run)
    current = await OrchestrationProcessService().get_current(
        db_session, goal.id, "effectiveness_review"
    )
    decisions = await OrchestrationAuthorityDecisionService().list_decisions(
        db_session, goal.id
    )
    memory = await OrchestrationMemoryService().get_section(
        db_session, goal.project_id, goal.id, "effectiveness_review"
    )

    assert summary == {
        "process_type": "effectiveness_review",
        "status": "waiting_decision",
        "questions_created": 1,
        "recommended_disposition": "pause",
    }
    assert current.status == "waiting_decision"
    assert [trigger["name"] for trigger in current.input_snapshot["triggers"]] == ["inactivity"]
    assert [trigger["name"] for trigger in current.outputs["triggers"]] == ["inactivity"]
    assert current.outputs["recommended_disposition"] == "pause"
    assert current.outputs["run_id"] == str(run.id)
    assert current.outputs["decision_status"] == "pending"
    assert current.outputs["selected_disposition"] is None
    assert current.outputs["decision_reason"] is None
    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.id == uuid.UUID(current.outputs["decision_id"])
    assert decision.source_process_run_id == current.id
    assert decision.run_id == run.id
    assert decision.decision_key == f"effectiveness_review:disposition:{current.id}"
    assert [option["key"] for option in decision.options] == [
        "continue",
        "revise",
        "split",
        "pause",
    ]
    assert decision.recommendation == "pause"
    assert decision.authority == "manager"
    assert decision.authority_agent_id == test_agent.id
    assert json.loads(memory.body)["decision"] == {
        "id": str(decision.id),
        "reason": None,
        "selected_disposition": None,
        "status": "pending",
    }

    retry = await process.advance(db_session, goal, run)
    assert retry["status"] == "waiting_decision"
    assert retry["questions_created"] == 0
    assert len(await OrchestrationAuthorityDecisionService().list_decisions(db_session, goal.id)) == 1
    assert stub.calls == 1


@pytest.mark.asyncio
async def test_clean_pre_completion_starts_and_completes_in_one_advance(
    db_session, test_project
):
    goal, run = await _goal_run(db_session, test_project)
    goal.success_criteria = [{"description": "Observable result"}]
    goal.authority_model = "no_manager"
    db_session.add_all(
        (
            OrchestrationGate(
                run_id=run.id,
                success_criterion_key="criterion",
                gate_type="evidence",
                required_evidence={},
                status="accepted",
            ),
            OrchestrationGate(
                run_id=run.id,
                success_criterion_key="summary",
                gate_type="final_summary_accepted",
                required_evidence={},
                status="open",
            ),
        )
    )
    await db_session.flush()

    stub = _EffectivenessAnalyzerStub("continue")
    summary = await orchestration_effectiveness_review.EffectivenessReviewProcess(stub).advance(
        db_session, goal, run
    )
    current = await OrchestrationProcessService().get_current(
        db_session, goal.id, "effectiveness_review"
    )

    assert summary == {
        "process_type": "effectiveness_review",
        "status": "completed",
        "questions_created": 0,
        "recommended_disposition": "continue",
    }
    assert current.status == "completed"
    assert [trigger["name"] for trigger in current.outputs["triggers"]] == ["pre_completion"]
    assert current.outputs["recommendations"] == [
        {"disposition": "continue", "detail": stub.rationale}
    ]
    assert current.outputs["decision_id"] is None
    assert await OrchestrationAuthorityDecisionService().list_decisions(db_session, goal.id) == []
    assert stub.calls == 1
    assert run.active_blockers == []


@pytest.mark.asyncio
async def test_manual_running_review_adds_current_automatic_evidence_and_completes(
    db_session, test_project
):
    goal, run = await _goal_run(db_session, test_project)
    goal.success_criteria = [{"description": "Observable result"}]
    goal.authority_model = "no_manager"
    db_session.add(
        OrchestrationGate(
            run_id=run.id,
            success_criterion_key="criterion",
            gate_type="evidence",
            required_evidence={},
            status="accepted",
        )
    )
    await db_session.flush()
    current = await OrchestrationProcessService().start_process(
        db_session,
        goal.id,
        process_type="effectiveness_review",
        trigger_reason="human requested: inspect progress",
        run_id=run.id,
    )

    stub = _EffectivenessAnalyzerStub("continue")
    summary = await orchestration_effectiveness_review.EffectivenessReviewProcess(stub).advance(
        db_session, goal, run
    )

    assert summary["status"] == "completed"
    assert summary["recommended_disposition"] == "continue"
    assert current.status == "completed"
    assert [trigger["name"] for trigger in current.outputs["triggers"]] == [
        "manual",
        "pre_completion",
    ]
    assert current.outputs["triggers"][0] == {
        "name": "manual",
        "token": f"manual:{current.id}",
        "detail": "human requested: inspect progress",
    }
    assert stub.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("decision_status", "selected", "reason"),
    (
        ("answered", "continue", "Current plan remains effective."),
        ("cancelled", None, "no longer needed"),
        ("expired", None, None),
    ),
)
async def test_waiting_review_consumes_its_terminal_linked_decision_once_even_when_run_paused(
    db_session, test_project, test_user, decision_status, selected, reason
):
    goal, run = await _goal_run(db_session, test_project)
    goal.success_criteria = [{"description": "Observable result"}]
    goal.authority_model = "no_manager"
    run.started_at = _utcnow() - timedelta(hours=25)
    stub = _EffectivenessAnalyzerStub("pause")
    process = orchestration_effectiveness_review.EffectivenessReviewProcess(stub)
    await process.advance(db_session, goal, run)
    current = await OrchestrationProcessService().get_current(
        db_session, goal.id, "effectiveness_review"
    )
    decision_service = OrchestrationAuthorityDecisionService()
    decision = (await decision_service.list_decisions(db_session, goal.id))[0]
    if decision_status == "answered":
        await decision_service.answer_decision(
            db_session,
            decision,
            selected_option="continue",
            reason=reason,
            decided_by_user_id=test_user.id,
        )
    elif decision_status == "cancelled":
        await decision_service.cancel_decision(db_session, decision, reason="no longer needed")
    else:
        decision.status = "expired"
        await db_session.flush()
    run.status = "paused"

    summary = await process.advance(db_session, goal, run)
    memory = await OrchestrationMemoryService().get_section(
        db_session, goal.project_id, goal.id, "effectiveness_review"
    )

    assert summary["status"] == "completed"
    assert summary["decision_id"] == str(decision.id)
    assert summary["decision_status"] == decision_status
    assert summary["selected_disposition"] == selected
    assert current.status == "completed"
    assert current.outputs["decision_status"] == decision_status
    assert current.outputs["selected_disposition"] == selected
    assert current.outputs["decision_reason"] == reason
    assert json.loads(memory.body)["decision"] == {
        "id": str(decision.id),
        "reason": reason,
        "selected_disposition": selected,
        "status": decision_status,
    }
    assert (await process.advance(db_session, goal, run))["status"] == "completed"
    assert stub.calls == 1


@pytest.mark.asyncio
async def test_completed_review_with_new_evidence_starts_and_processes_successor(
    db_session, test_project
):
    goal, run = await _goal_run(db_session, test_project)
    goal.success_criteria = [{"description": "Observable result"}]
    goal.authority_model = "no_manager"
    first_idle = _utcnow() - timedelta(hours=25)
    run.started_at = first_idle
    process_service = OrchestrationProcessService()
    previous = await process_service.start_process(
        db_session,
        goal.id,
        process_type="effectiveness_review",
        trigger_reason="automatic: inactivity",
        run_id=run.id,
        input_snapshot={"triggers": []},
    )
    await process_service.complete_process(
        db_session,
        previous,
        outputs={
            "triggers": [
                {
                    "name": "inactivity",
                    "token": first_idle.isoformat(),
                    "detail": "Previous inactivity",
                }
            ],
            "recommended_disposition": "pause",
        },
    )

    stub = _EffectivenessAnalyzerStub("pause")
    process = orchestration_effectiveness_review.EffectivenessReviewProcess(stub)

    same = await process.advance(db_session, goal, run)
    assert same["status"] == "completed"
    assert stub.calls == 0, "no-new-trigger-evidence short-circuit must not touch the analyzer"
    assert await process_service.list_process_runs(
        db_session, goal.id, process_type="effectiveness_review"
    ) == [previous]

    run.started_at = first_idle - timedelta(hours=24)
    await db_session.flush()
    summary = await process.advance(db_session, goal, run)
    assert stub.calls == 1
    current = await process_service.get_current(db_session, goal.id, "effectiveness_review")
    process_runs = await process_service.list_process_runs(
        db_session, goal.id, process_type="effectiveness_review"
    )

    assert summary["status"] == "waiting_decision"
    assert summary["recommended_disposition"] == "pause"
    assert current.id != previous.id
    assert current.status == "waiting_decision"
    assert current.input_snapshot["triggers"][0]["token"] == run.started_at.isoformat()
    assert len(process_runs) == 2
    assert previous.superseded_by_id == current.id


async def _stub_tick_baseline(monkeypatch):
    from huddleroom.services import orchestration_service

    async def complete(*_args, **_kwargs):
        return {"status": "completed"}

    for process in (
        orchestration_service.GoalDefinitionProcess,
        orchestration_service.ManagerSelectionProcess,
        orchestration_service.AgentDefinitionReviewProcess,
        orchestration_service.TeamHierarchyProcess,
    ):
        monkeypatch.setattr(process, "advance", complete)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("review_status", "finish_expected"),
    [
        ("idle", True),
        ("running", True),
        ("completed", True),
        ("skipped", True),
        ("waiting_decision", False),
    ],
)
async def test_tick_runs_effectiveness_after_recovery_and_only_waiting_blocks_finish(
    db_session,
    test_project,
    monkeypatch,
    review_status,
    finish_expected,
):
    from huddleroom.services import orchestration_service
    from huddleroom.services.orchestration_llm_decision_adapter import (
        OrchestrationDecisionAdapter,
        OrchestrationDecisionAdapterResult,
    )
    from tests.test_orchestration_debug import _seed_terminal

    goal, run = await _goal_run(db_session, test_project)
    # can_finish (final_summary/closeout/completion) now gates on
    # run.phase == "authorized" (Task 3); this test's finish_expected=True
    # cases require the completion tail to run.
    run.phase = "authorized"
    await db_session.flush()
    # tick()'s own baseline_ready is computed from the (mocked, below) process
    # .advance() calls, but the authorized-execution loop's request_llm_decision
    # separately gates on real DB process rows via _ensure_baseline_processes_ready
    # -- seed those directly (mirrors _seed_baseline_terminal_run) so that gate
    # doesn't 409 out from under the mocked-terminal tick.
    await _seed_terminal(db_session, goal.id, "goal_definition", run.id)
    await _seed_terminal(db_session, goal.id, "manager_selection", run.id)
    await _seed_terminal(db_session, goal.id, "agent_definition_review", run.id, terminal="skipped")
    await _seed_terminal(db_session, goal.id, "team_hierarchy", run.id, terminal="skipped")

    async def fake_decide(self, context, *, project=None, goal=None):
        return OrchestrationDecisionAdapterResult(
            input_snapshot=dict(context),
            llm_output={"raw_content": None},
            parsed_decision={"action_type": "noop", "reason": "test stub"},
        )

    monkeypatch.setattr(OrchestrationDecisionAdapter, "decide", fake_decide)
    await _stub_tick_baseline(monkeypatch)
    calls = []
    service = OrchestrationService()

    async def validate(*_args, **_kwargs):
        calls.append("validate")
        return 0

    async def recover(*_args, **_kwargs):
        calls.append("recover")
        return 0

    async def review(*_args, **_kwargs):
        calls.append("effectiveness")
        return {
            "process_type": "effectiveness_review",
            "status": review_status,
            "questions_created": 0,
            "recommended_disposition": None,
        }

    async def final_summary_ready(*_args, **_kwargs):
        calls.append("final_summary")
        return False

    async def closeout_preconditions(*_args, **_kwargs):
        calls.append("closeout_preconditions")
        return {}

    async def closeout(*_args, **_kwargs):
        calls.append("closeout")
        return {"status": "completed", "completion_authorized": True}

    async def completion_ready(*_args, **_kwargs):
        calls.append("completion")
        return False

    monkeypatch.setattr(service, "validate_open_gates", validate)
    monkeypatch.setattr(service, "recover_run", recover)
    monkeypatch.setattr(
        orchestration_effectiveness_review.EffectivenessReviewProcess,
        "advance",
        review,
    )
    monkeypatch.setattr(
        service,
        "_run_ready_for_final_summary_request",
        final_summary_ready,
    )
    monkeypatch.setattr(service, "_closeout_preconditions_manifest", closeout_preconditions)
    monkeypatch.setattr(
        orchestration_service.GoalCloseoutProcess,
        "advance",
        closeout,
    )
    monkeypatch.setattr(service, "_run_ready_for_completion", completion_ready)

    result = await service.tick(db_session, run.id)

    assert calls == [
        "validate",
        "recover",
        "effectiveness",
        *(
            ["final_summary", "closeout_preconditions", "closeout", "completion"]
            if finish_expected
            else []
        ),
    ]
    assert result["effectiveness_review_process"]["status"] == review_status


@pytest.mark.asyncio
async def test_recovery_that_blocks_run_skips_effectiveness_review(
    db_session,
    test_project,
    test_agent,
    monkeypatch,
):
    goal, run = await _goal_run(db_session, test_project)
    goal.success_criteria = [{"description": "Observable result"}]
    goal.authority_model = "no_manager"
    task = Task(
        project_id=test_project.id,
        title="Repeatedly failing work",
        status="failed",
        assigned_to=test_agent.id,
        metadata_={"orchestration": {"run_id": str(run.id)}},
    )
    db_session.add(task)
    await db_session.flush()
    run.plan_state = {
        "expanded_items": [
            {"task_id": str(task.id), "gate_id": str(uuid.uuid4())}
        ]
    }
    db_session.add_all(
        Session(
            task_id=task.id,
            agent_id=test_agent.id,
            project_id=test_project.id,
            adapter_type="api",
            status="failed",
            metadata_={},
            origin="auto",
        )
        for _ in range(3)
    )
    await db_session.flush()
    await _stub_tick_baseline(monkeypatch)
    stub = _EffectivenessAnalyzerStub("revise")
    monkeypatch.setattr(EffectivenessAnalyzer, "review_request", stub.review_request)

    result = await OrchestrationService().tick(db_session, run.id)

    assert result["status"] == "blocked"
    assert result["recoveries_created"] == 1
    assert OrchestrationService().run_condition(goal, run) == "needs_attention"
    assert result["effectiveness_review_process"] is None
    current = await OrchestrationProcessService().get_current(
        db_session, goal.id, "effectiveness_review"
    )
    assert current is None
    assert stub.calls == 0


@pytest.mark.asyncio
async def test_force_tick_preflight_autobegin_commits_paused_settlement(
    concurrent_sessions, tmp_path, monkeypatch,
):
    from httpx import ASGITransport, AsyncClient

    from huddleroom.database import get_db
    from huddleroom.main import create_app
    from huddleroom.models.project import Project

    writer, reader = concurrent_sessions
    project = Project(
        name=f"effectiveness-route-persistence-{uuid.uuid4()}",
        description="Route persistence boundary",
        workspace_path=str(tmp_path),
        config={},
    )
    writer.add(project)
    await writer.flush()
    goal = OrchestrationGoal(
        project_id=project.id,
        objective="Persist route-triggered paused review settlement",
        success_criteria=[{"description": "Observable result"}],
        constraints={},
        budget={},
        authority_model="no_manager",
    )
    writer.add(goal)
    await writer.flush()
    run = OrchestrationRun(
        goal_id=goal.id,
        started_at=_utcnow() - timedelta(hours=25),
    )
    writer.add(run)
    await writer.commit()

    # Non-"continue" disposition: the review must park behind a decision (rather
    # than complete immediately) so the route-triggered tick below has a pending
    # decision to settle once it's expired.
    stub = _EffectivenessAnalyzerStub("pause")
    monkeypatch.setattr(EffectivenessAnalyzer, "review_request", stub.review_request)

    await orchestration_effectiveness_review.EffectivenessReviewProcess().advance(
        writer, goal, run
    )
    current = await OrchestrationProcessService().get_current(
        writer, goal.id, "effectiveness_review"
    )
    decision = (
        await OrchestrationAuthorityDecisionService().list_decisions(writer, goal.id)
    )[0]
    await writer.commit()
    decision.status = "expired"
    run.status = "paused"
    await writer.commit()

    app = create_app()

    async def override_get_db():
        yield writer

    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as route_client:
        response = await route_client.post(
            f"/api/v1/projects/{project.id}/orchestration/runs/{run.id}/tick"
        )

    persisted = await reader.get(OrchestrationProcessRun, current.id)
    assert response.status_code == 200, response.text
    assert response.json()["effectiveness_review_process"]["status"] == "completed"
    assert persisted.status == "completed"
    assert stub.calls == 1


@pytest.mark.asyncio
async def test_analyzer_failure_creates_effectiveness_review_analyzer_error_blocker(
    db_session, test_project
):
    goal, run = await _goal_run(db_session, test_project)
    goal.success_criteria = [{"description": "Observable result"}]
    goal.authority_model = "no_manager"
    run.started_at = _utcnow() - timedelta(hours=25)
    stub = _EffectivenessAnalyzerFailThenSucceedStub()
    process = orchestration_effectiveness_review.EffectivenessReviewProcess(stub)

    result = await process.advance(db_session, goal, run)

    assert result["status"] == "running"
    assert result["retryable"] is True
    assert result["error"] == "RuntimeError: provider unavailable"
    current = await OrchestrationProcessService().get_current(
        db_session, goal.id, "effectiveness_review"
    )
    assert current.status == "running"
    checkpoint = current.outputs["_lm_retry"]
    assert checkpoint["kind"] == "effectiveness_review"
    warning_id = checkpoint["warning_id"]
    assert isinstance(warning_id, str)

    assert len(run.active_blockers) == 1
    blocker = run.active_blockers[0]
    assert blocker["kind"] == "effectiveness_review_analyzer_error"
    assert blocker["warning_id"] == warning_id

    warnings = await OrchestrationWarningService().list_warnings(
        db_session, goal.id, active_only=True
    )
    assert len(warnings) == 1
    assert warnings[0].warning_type == "effectiveness_review_analyzer_error"
    assert str(warnings[0].id) == warning_id
    assert stub.calls == 1


@pytest.mark.asyncio
async def test_retry_success_clears_effectiveness_review_blocker_and_resolves_warning(
    db_session, test_project
):
    goal, run = await _goal_run(db_session, test_project)
    goal.success_criteria = [{"description": "Observable result"}]
    goal.authority_model = "no_manager"
    run.started_at = _utcnow() - timedelta(hours=25)
    stub = _EffectivenessAnalyzerFailThenSucceedStub(disposition="pause")
    process = orchestration_effectiveness_review.EffectivenessReviewProcess(stub)
    await process.advance(db_session, goal, run)
    current = await OrchestrationProcessService().get_current(
        db_session, goal.id, "effectiveness_review"
    )
    warning_id = current.outputs["_lm_retry"]["warning_id"]
    assert len(run.active_blockers) == 1

    stub.should_fail = False
    result = await process.retry_failed(db_session, goal, run, current)

    assert result["status"] == "waiting_decision"
    assert result["recommended_disposition"] == "pause"
    assert run.active_blockers == []

    active_warnings = await OrchestrationWarningService().list_warnings(
        db_session, goal.id, active_only=True
    )
    assert not any(str(w.id) == warning_id for w in active_warnings)
    all_warnings = await OrchestrationWarningService().list_warnings(
        db_session, goal.id, active_only=False
    )
    resolved = next(w for w in all_warnings if str(w.id) == warning_id)
    assert resolved.active is False
    assert resolved.resolved_by == "orchestrator:effectiveness_review"

    assert "_lm_retry" not in current.outputs
    assert "error" not in current.outputs
    assert "retryable" not in current.outputs
    assert stub.calls == 2


@pytest.mark.asyncio
async def test_repeated_analyzer_failures_do_not_duplicate_blocker_or_warning(
    db_session, test_project
):
    goal, run = await _goal_run(db_session, test_project)
    goal.success_criteria = [{"description": "Observable result"}]
    goal.authority_model = "no_manager"
    run.started_at = _utcnow() - timedelta(hours=25)
    stub = _EffectivenessAnalyzerFailThenSucceedStub()
    process = orchestration_effectiveness_review.EffectivenessReviewProcess(stub)
    await process.advance(db_session, goal, run)
    current = await OrchestrationProcessService().get_current(
        db_session, goal.id, "effectiveness_review"
    )

    await process.retry_failed(db_session, goal, run, current)

    assert len(run.active_blockers) == 1
    warnings = await OrchestrationWarningService().list_warnings(
        db_session, goal.id, active_only=True
    )
    assert len(warnings) == 1
    assert stub.calls == 2
