"""SPR 180/184: trigger activation and the legacy-bypass cutover.

Fake completions / synthetic rows only. The recovery analyzer is always stubbed so no tick in this
file can reach a provider once the writer wires tick-time diagnosis.

Interface assumptions (from the spec, brief and carry-ins):
- A trigger is an ``OrchestrationAction`` of type ``recovery_trigger`` whose request is the sealed
  ``RecoveryTriggerV1`` (flat ``source_kind``/``source_id``/``blocker_code``/``retry_target``).
- ``recover_blocker`` is a dispatchable, non-mutating decision: source ``reported_blocker`` = its decision.
- A timer-only noop with no relevant live producer becomes source ``eventless_noop`` = its decision.
- ``plan_author_unavailable`` becomes source ``planning_preflight`` with retry target ``planning``.
- The tick diagnoses reserved triggers after committing them and drives ``resume`` for approved
  proposals and due recovery-owned waits (carry-in 1); supervision never fires the generic
  attention/noop fallback for an ``owner.type == "recovery"`` wait (carry-in 2).
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import func, select

from huddleroom.models.agent import Agent
from huddleroom.models.orchestration import (
    OrchestrationAction, OrchestrationDecision, OrchestrationEvidence, OrchestrationGoal, OrchestrationRun,
    OrchestrationWait,
)
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision, OrchestrationWarning
from huddleroom.models.project import Project
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.schemas.orchestration_recovery import (
    CapabilityGapAssessmentV1, CorrectiveDisposition, RecoveryTriggerV1, TransientAssessmentV1,
)
from huddleroom.services.orchestration_blocker_recovery import (
    OrchestrationBlockerRecoveryService, RecoveryDiagnosisFailed,
)
from huddleroom.services.orchestration_decision_dispatcher import OrchestrationDecisionDispatcher
from huddleroom.services.orchestration_recovery_analyzer import RecoveryAnalyzer
from huddleroom.services.orchestration_service import OrchestrationService
from tests.test_orchestration_roster_mutation import sessions  # noqa: F401 -- fixture re-exported

pytestmark = pytest.mark.usefixtures(
    "safe_goal_analysis", "safe_agent_definition_review", "safe_effectiveness_review_continue", "no_live_decisions",
)


@pytest.fixture
def no_live_decisions(stub_decision):
    """Every tick decision is a fake; tests that need a specific decision call ``stub_decision`` again."""
    stub_decision(lambda _ctx: {"action_type": "ask_human", "question": "Proceed?", "reason": "Unsure."})


# --------------------------------------------------------------------------- helpers


def _gap(source_id: uuid.UUID) -> CapabilityGapAssessmentV1:
    return CapabilityGapAssessmentV1.model_validate({
        "kind": "capability_gap",
        "diagnosis": "No registered tool can repair this blocker.",
        "evidence": [{"label": "The run is blocked.", "value": {"status": "blocked"}, "source_kind": "run",
                      "source_id": str(source_id)}],
        "missing_capability": "A human decision about the blocked work.",
        "required_human_action": "Decide how the blocked work should continue.",
        "wake_after": {"kind": "human_decision", "decision": "Owner decided"},
    })


@pytest.fixture
def analyzer_calls(monkeypatch):
    """Stub the recovery LM for every service instance: always a capability gap, never a provider."""
    calls: list[dict] = []

    async def analyze(self, snapshot, *, project_id, evidence_query, on_raw=None):
        calls.append({"snapshot": snapshot, "project_id": project_id})
        return _gap(uuid.uuid4())

    monkeypatch.setattr(RecoveryAnalyzer, "analyze", analyze)
    return calls


async def _actions(db, run_id, action_type):
    return list((await db.scalars(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run_id, OrchestrationAction.action_type == action_type,
    ).order_by(OrchestrationAction.created_at, OrchestrationAction.id)
        .execution_options(populate_existing=True))).all())


def _blocker_code(trigger: OrchestrationAction) -> str:
    request = trigger.request or {}
    return (request.get("source") or request)["blocker_code"]


async def _decision(db, run, parsed, *, decision_type="coordination", snapshot=None):
    row = OrchestrationDecision(run_id=run.id, decision_type=decision_type, input_snapshot=snapshot or {},
                                parsed_decision=parsed, validator_status="accepted")
    db.add(row)
    await db.flush()
    return row


async def _authorized_goal(db, project):
    goal = OrchestrationGoal(project_id=project.id, objective="Recovery activation", status="active")
    db.add(goal)
    await db.flush()
    run = OrchestrationRun(goal_id=goal.id, status="running", phase="authorized", plan_state={"status": "accepted"})
    db.add(run)
    await db.flush()
    return goal, run


def _trigger(goal_id, run_id, *, source_kind="planning_preflight", source_id=None, retry=None,
             blocker_code="plan_author_unavailable") -> RecoveryTriggerV1:
    return RecoveryTriggerV1.model_validate({
        "schema_version": 1, "goal_id": str(goal_id), "run_id": str(run_id), "source_kind": source_kind,
        "source_id": str(source_id or run_id), "blocker_code": blocker_code,
        "summary": "Planning cannot start.", "source_fingerprint": "a" * 64,
        "retry_target": retry or {"kind": "planning"},
    })


# --------------------------------------------------------------------------- carry-in 6


@pytest.mark.asyncio
async def test_latest_decision_for_run_ignores_recovery_exchange_records(db_session, test_project):
    _goal, run = await _authorized_goal(db_session, test_project)
    ordinary = await _decision(db_session, run, {"action_type": "noop"})
    ordinary.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    recovery = await _decision(db_session, run, {}, decision_type="recovery")
    recovery.created_at = datetime(2026, 1, 2, tzinfo=timezone.utc)
    await db_session.flush()

    latest = await OrchestrationService()._latest_decision_for_run(db_session, run.id)

    assert latest is not None and latest.id == ordinary.id


# --------------------------------------------------------------------------- carry-in 2


@pytest.mark.asyncio
async def test_supervision_never_fires_generic_fallback_for_a_due_recovery_owned_wait(db_session, test_project):
    goal, run = await _authorized_goal(db_session, test_project)
    wait = OrchestrationWait(
        run_id=run.id, wait_key=f"recovery:{uuid.uuid4()}:delegation", owner={"type": "recovery", "id": str(uuid.uuid4())},
        awaited_event={"event_type": "task.status_changed", "matcher": {"task_id": str(uuid.uuid4())}},
        due_recheck_at=datetime.now(timezone.utc) - timedelta(minutes=5),
        fallback={"action_type": "attention", "expected_result": "Remediation finished.", "recheck_seconds": 60},
    )
    db_session.add(wait)
    await db_session.flush()

    outcome = await OrchestrationService().supervision.reconcile_local(db_session, goal, run, allow_release=False)

    assert outcome.get("outcome") != "due_fallback", outcome
    assert await _actions(db_session, run.id, "record_warning") == []
    assert await _actions(db_session, run.id, "noop") == []
    warnings = (await db_session.scalars(select(OrchestrationWarning).where(
        OrchestrationWarning.run_id == run.id, OrchestrationWarning.warning_type == "supervision_attention"))).all()
    assert warnings == []


# --------------------------------------------------------------------------- explicit recover_blocker


@pytest.mark.asyncio
async def test_explicit_recover_blocker_creates_one_reported_blocker_lineage_without_mutation(
    db_session, test_project, analyzer_calls,
):
    goal, run = await _authorized_goal(db_session, test_project)
    gate_task = Task(project_id=test_project.id, title="Blocked", status="blocked",
                     metadata_={"orchestration": {"run_id": str(run.id)}})
    db_session.add(gate_task)
    await db_session.flush()
    decision = await _decision(db_session, run, {
        "action_type": "recover_blocker", "reason": "The planner lacks an accepted planning function.",
        "evidence_refs": [{"kind": "task", "id": str(gate_task.id)}],
    })
    actions_before = await db_session.scalar(select(func.count(OrchestrationAction.id)))
    dispatcher = OrchestrationDecisionDispatcher(OrchestrationService())

    first = await dispatcher.dispatch(db_session, run, decision)
    await dispatcher.dispatch(db_session, run, decision)  # replay

    assert first is not None
    triggers = await _actions(db_session, run.id, "recovery_trigger")
    assert len(triggers) == 1
    request = triggers[0].request
    assert request["source_kind"] == "reported_blocker"
    assert request["source_id"] == str(decision.id)
    assert request["retry_target"] == {"kind": "decision_loop"}
    # Non-mutating: only the decision's own record and the trigger are new; the task is untouched.
    new_rows = await db_session.scalar(select(func.count(OrchestrationAction.id))) - actions_before
    assert new_rows <= 2
    await db_session.refresh(gate_task)
    assert gate_task.status == "blocked"
    assert analyzer_calls == []  # dispatch records the blocker; diagnosis happens outside the decision path


# --------------------------------------------------------------------------- planning preflight


async def _make_planner_ineligible(db, goal, run, planner):
    """Team hierarchy names ``planner`` for planning but its accepted eligibility excludes planning."""
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    service = OrchestrationProcessService()
    team = await service.get_current(db, goal.id, "team_hierarchy")
    review = await service.get_current(db, goal.id, "agent_definition_review")
    assert team is not None and review is not None
    team.outputs = {**(team.outputs or {}), "role_to_agent": {},
                    "assignments": [{"work_function": "planning", "agent_ref": str(planner.id)}]}
    review.outputs = {**(review.outputs or {}),
                      "eligibility": {**((review.outputs or {}).get("eligibility") or {}), str(planner.id): []}}
    await db.flush()


@pytest.mark.asyncio
async def test_plan_author_unavailable_creates_one_recovery_lineage(
    db_session, test_project, stub_decision, analyzer_calls,
):
    from huddleroom.services.orchestration_team_hierarchy import select_plan_author
    from tests.test_orchestration_runtime_e2e import _authorized_run

    planner = Agent(name=f"planner-{uuid.uuid4()}", role="planner", provider="openai", model="gpt-4o-mini",
                    adapter_type="api", capabilities=["planning"], config={}, is_active=True)
    db_session.add(planner)
    await db_session.flush()
    stub_decision(lambda _ctx: {"action_type": "request_plan", "agent_id": str(planner.id),
                                "work_function": "planning", "scope": "Plan the goal."})
    service, goal, run = await _authorized_run(db_session, test_project)
    await _make_planner_ineligible(db_session, goal, run, planner)
    with pytest.raises(HTTPException) as preflight:
        await select_plan_author(db_session, goal, run)
    assert str(preflight.value.detail).startswith("plan_author_unavailable"), "precondition"

    await service.tick(db_session, run.id)
    await service.tick(db_session, run.id)

    triggers = await _actions(db_session, run.id, "recovery_trigger")
    assert len(triggers) == 1
    assert _blocker_code(triggers[0]) == "plan_author_unavailable"
    assert triggers[0].request["source_kind"] == "planning_preflight"
    assert triggers[0].request["retry_target"]["kind"] == "planning"
    # The blocker reaches a recovery outcome (proposal or capability card), not a bare 409.
    recovery_cards = (await db_session.scalars(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.run_id == run.id,
        OrchestrationAuthorityDecision.decision_key.startswith("recovery")))).all()
    assert len(await _actions(db_session, run.id, "recovery_proposal")) == 1
    assert len(recovery_cards) == 1


# --------------------------------------------------------------------------- eventless noop


@pytest.mark.asyncio
async def test_timer_only_noop_without_live_producer_creates_one_eventless_lineage(
    db_session, test_project, stub_decision, analyzer_calls,
):
    from tests.test_orchestration_runtime_e2e import _authorized_run

    stub_decision(lambda _ctx: {"action_type": "noop", "reason": "Waiting for things to improve.",
                                "wake_when": {"recheck_after_seconds": 300, "expected_result": "Something changes."}})
    service, _goal, run = await _authorized_run(db_session, test_project)

    await service.tick(db_session, run.id)

    triggers = await _actions(db_session, run.id, "recovery_trigger")
    assert len(triggers) == 1
    request = triggers[0].request
    assert request["source_kind"] == "eventless_noop"
    noop_decision = await db_session.get(OrchestrationDecision, uuid.UUID(request["source_id"]))
    assert noop_decision is not None and noop_decision.parsed_decision["action_type"] == "noop"
    assert request["retry_target"] == {"kind": "decision_loop"}


@pytest.mark.asyncio
async def test_noop_waiting_on_a_relevant_pending_human_decision_stays_a_wait(
    db_session, test_project, analyzer_calls,
):
    """Guard against over-triggering: a real pending owner decision is a live producer, so the noop
    (decided from the server-built context) stays a wait and starts no recovery lineage."""
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
    from tests.test_orchestration_runtime_e2e import _authorized_run

    service, goal, run = await _authorized_run(db_session, test_project)
    pending = await OrchestrationAuthorityDecisionService().create_pending(
        db_session, goal.id, decision_key=f"owner:{uuid.uuid4()}", title="Choose scope", question="Which scope?",
        authority="human", options=["a", "b"], run_id=run.id,
    )
    context = await service._decision_context(db_session, goal, run)
    decision = await _decision(db_session, run, {
        "action_type": "noop", "reason": "Waiting for the owner's answer.",
        "wake_when": {"events": [{"event_type": "authority.decision_resolved",
                                  "matcher": {"decision_id": str(pending.id)}}],
                      "recheck_after_seconds": None, "expected_result": "The owner answers."},
    }, snapshot=context)

    action = await service._dispatch_execution_decision(db_session, run, decision)

    assert action is not None and action.action_type == "noop"
    assert await _actions(db_session, run.id, "recovery_trigger") == []


@pytest.mark.asyncio
async def test_dispatched_timer_only_noop_without_relevant_producer_creates_one_trigger(
    db_session, test_project, analyzer_calls,
):
    from tests.test_orchestration_runtime_e2e import _authorized_run

    service, goal, run = await _authorized_run(db_session, test_project)
    context = await service._decision_context(db_session, goal, run)
    decision = await _decision(db_session, run, {
        "action_type": "noop", "reason": "Waiting.",
        "wake_when": {"events": [], "recheck_after_seconds": 300, "expected_result": "Something changes."},
    }, snapshot=context)

    await service._dispatch_execution_decision(db_session, run, decision)
    await service._dispatch_execution_decision(db_session, run, decision)  # replay

    triggers = await _actions(db_session, run.id, "recovery_trigger")
    assert len(triggers) == 1
    assert triggers[0].request["source_kind"] == "eventless_noop"
    assert triggers[0].request["source_id"] == str(decision.id)


# --------------------------------------------------------------------------- transient assessment


@pytest.mark.asyncio
@pytest.mark.parametrize("producer_status", ["ready", "done"])
async def test_transient_assessment_is_accepted_only_with_a_live_producer(db_session, test_project, producer_status):
    """producer check: a transient verdict naming a live, relevant producer becomes a wait, not a card."""
    from huddleroom.models.orchestration import OrchestrationGate

    goal, run = await _authorized_goal(db_session, test_project)
    gate = OrchestrationGate(run_id=run.id, success_criterion_key=str(uuid.uuid4()), gate_type="work_completed",
                             status="accepted" if producer_status == "done" else "open")
    db_session.add(gate)
    await db_session.flush()
    producer = Task(project_id=test_project.id, title="Producer", status=producer_status,
                    metadata_={"orchestration": {"run_id": str(run.id), "gate_id": str(gate.id)}})
    db_session.add(producer)
    await db_session.flush()
    blocked = Task(project_id=test_project.id, title="Blocked", status="blocked", depends_on=[str(producer.id)],
                   metadata_={"orchestration": {"run_id": str(run.id)}})
    db_session.add(blocked)
    await db_session.flush()
    source = OrchestrationAction(run_id=run.id, idempotency_key=f"run:{run.id}:kind:create_delegation_task:{uuid.uuid4()}",
                                 action_type="create_delegation_task", request={"task_id": str(blocked.id)},
                                 status="failed", error="Task is blocked", target_type="task", target_id=blocked.id)
    db_session.add(source)
    await db_session.flush()
    transient = TransientAssessmentV1.model_validate({
        "kind": "transient", "diagnosis": "The dependency is still being produced.",
        "evidence": [{"label": "Producer status", "value": {"status": producer_status}, "source_kind": "task",
                      "source_id": str(producer.id)}],
        "wake": {"events": [{"event_type": "task.status_changed", "matcher": {"task_id": str(producer.id)}}],
                 "expected_result": "The producer finishes."},
    })

    class Transient:
        async def analyze(self, snapshot, *, project_id, evidence_query, on_raw=None):
            return transient

    svc = OrchestrationBlockerRecoveryService(analyzer=Transient())
    trigger = await svc.create_trigger(db_session, _trigger(
        goal.id, run.id, source_kind="failed_action", source_id=source.id,
        retry={"kind": "action", "source_action_id": str(source.id)}, blocker_code="dependency_pending"))

    if producer_status == "ready":
        await svc.diagnose(db_session, trigger.id)
        await db_session.refresh(trigger)
        assert trigger.status == "completed"
        assert await _actions(db_session, run.id, "recovery_proposal") == []
        cards = (await db_session.scalars(select(OrchestrationAuthorityDecision).where(
            OrchestrationAuthorityDecision.run_id == run.id))).all()
        assert cards == []
    else:
        with pytest.raises(RecoveryDiagnosisFailed):
            await svc.diagnose(db_session, trigger.id)


# --------------------------------------------------------------------------- carry-in 4: healthy work


@pytest.mark.asyncio
async def test_healthy_follow_up_on_a_recovery_created_plan_task_stays_ordinary(db_session, test_project):
    """A plan task produced by an approved recovery retry is ordinary once healthy: revising its plan is
    normal work, not a new corrective fix that needs another approval."""
    _goal, run = await _authorized_goal(db_session, test_project)
    plan_task = Task(project_id=test_project.id, title="Plan", status="done",
                     metadata_={"orchestration": {"run_id": str(run.id), "work_function": "planning"}})
    db_session.add(plan_task)
    await db_session.flush()
    db_session.add(OrchestrationAction(
        run_id=run.id, idempotency_key=f"recovery:{uuid.uuid4()}:retry:1:effect", action_type="request_plan",
        request={}, status="completed", target_type="task", target_id=plan_task.id,
    ))
    await db_session.flush()
    service = OrchestrationService()
    request = service.canonical_decision_request(
        "request_plan_revision",
        {"action_type": "request_plan_revision", "plan_task_id": str(plan_task.id), "revision_request": "Split item 2."},
        run_id=run.id,
    )

    disposition = await service.classify_corrective_action(db_session, run, "request_plan_revision", request, source="model")

    assert disposition == CorrectiveDisposition.ORDINARY


# --------------------------------------------------------------------------- tick wiring (carry-in 1)


@pytest.mark.asyncio
async def test_tick_with_failed_task_creates_one_trigger_and_never_retries(db_session, test_project, analyzer_calls):
    from tests.test_orchestration_basic_recovery import (
        _actions as basic_actions, _add_failed_session, _agent, _make_gate, _make_orchestrated_task, _make_run,
    )

    developer = _agent("developer", "developer", ["implementation"])
    db_session.add(developer)
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)
    task = await _make_orchestrated_task(db_session, test_project.id, run.id, gate.id, developer.id)
    await _add_failed_session(db_session, task, developer.id)

    await service.tick(db_session, run.id)
    await service.tick(db_session, run.id)

    assert await basic_actions(db_session, run.id, "retry_task") == []
    assert await basic_actions(db_session, run.id, "reassign_task") == []
    assert len(await basic_actions(db_session, run.id, "recovery_trigger")) == 1
    assert await db_session.scalar(select(func.count(Session.id)).where(Session.task_id == task.id)) == 1


async def _committed_authorized_run(sessions, tmp_path):  # noqa: F811 -- fixture name reused as parameter
    from tests.test_orchestration_runtime_e2e import _authorized_run

    async with sessions() as s:
        workspace = tmp_path / f"ws-{uuid.uuid4().hex[:6]}"
        workspace.mkdir()
        project = Project(name="Activation", workspace_path=str(workspace.resolve()), config={})
        s.add(project)
        await s.flush()
        _service, goal, run = await _authorized_run(s, project)
        await s.commit()
        return SimpleNamespace(project_id=project.id, goal_id=goal.id, run_id=run.id)


@pytest.mark.asyncio
async def test_tick_diagnoses_a_reserved_trigger_after_committing_it(
    sessions, tmp_path, monkeypatch, stub_decision,  # noqa: F811
):
    """Carry-in 1(a): the trigger is committed before the analyzer runs and no SQLite write lock is held
    across the analyzer call (an independent writer can commit while the analyzer is "thinking")."""
    stub_decision(lambda _ctx: {"action_type": "ask_human", "question": "Proceed?", "reason": "Unsure."})
    seeded = await _committed_authorized_run(sessions, tmp_path)
    async with sessions() as s:
        trigger = await OrchestrationBlockerRecoveryService().create_trigger(
            s, _trigger(seeded.goal_id, seeded.run_id))
        await s.commit()
        trigger_id = trigger.id
    observed: dict = {}

    async def analyze(self, snapshot, *, project_id, evidence_query, on_raw=None):
        async def independent_write():
            async with sessions() as other:
                row = await other.get(OrchestrationAction, trigger_id)
                observed["trigger_visible"] = row is not None
                other.add(OrchestrationWarning(goal_id=seeded.goal_id, run_id=seeded.run_id, warning_type="probe",
                                               severity="recommendation", message="independent writer"))
                await other.commit()
                observed["committed"] = True
        await asyncio.wait_for(independent_write(), timeout=5)
        return _gap(seeded.run_id)

    monkeypatch.setattr(RecoveryAnalyzer, "analyze", analyze)

    async with sessions() as s:
        await OrchestrationService().tick(s, seeded.run_id)

    assert observed == {"trigger_visible": True, "committed": True}
    async with sessions() as s:
        trigger = await s.get(OrchestrationAction, trigger_id)
        assert trigger.status == "completed"
        assert len(await _actions(s, seeded.run_id, "recovery_proposal")) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("wait_due", [True, False])
async def test_tick_drives_resume_for_approved_proposals_and_recovery_waits(
    db_session, test_project, monkeypatch, stub_decision, analyzer_calls, wait_due,
):
    """Carry-in 1(b)/(c) + 2: an approved non-terminal proposal is resumed every tick, including while a
    recovery-owned (delegated remediation) wait is open and while it is due."""
    from tests.test_orchestration_recovery_adapters import approved_call
    from tests.test_orchestration_runtime_e2e import _authorized_run

    planner = Agent(name=f"planner-{uuid.uuid4()}", role="planner", provider="openai", model="gpt-4o-mini",
                    adapter_type="api", capabilities=["planning"], config={}, is_active=True)
    db_session.add(planner)
    await db_session.flush()
    stub_decision(lambda _ctx: {"action_type": "ask_human", "question": "Proceed?", "reason": "Unsure."})
    service, goal, run = await _authorized_run(db_session, test_project)
    # recovery: the delegated remediation task has an assignee (an unassigned one is a separate
    # orphaned_ownership attention, not the recovery-wait fallback this test pins).
    task = Task(project_id=test_project.id, title="Delivery", status="ready", assigned_to=planner.id,
                metadata_={"orchestration": {"run_id": str(run.id)}})
    db_session.add(task)
    await db_session.flush()
    rows = SimpleNamespace(goal=goal, run=run, planner=planner, task=task)
    approved = await approved_call(db_session, rows, "orchestration.request_plan",
                                   {"agent_id": str(planner.id), "work_function": "planning", "scope": "goal"})
    db_session.add(OrchestrationWait(
        run_id=run.id, wait_key=f"recovery:{approved.proposal.proposal_id}:delegation",
        owner={"type": "recovery", "id": str(approved.wrapper.id)},
        awaited_event={"event_type": "task.status_changed", "matcher": {"task_id": str(task.id)}},
        due_recheck_at=datetime.now(timezone.utc) + (timedelta(minutes=-5) if wait_due else timedelta(hours=1)),
        fallback={"action_type": "attention", "expected_result": "Remediation finished.", "recheck_seconds": 60},
    ))
    await db_session.flush()
    resumed: list = []

    async def resume(self, db, proposal_action_id):
        resumed.append(proposal_action_id)
        return None

    monkeypatch.setattr(OrchestrationBlockerRecoveryService, "resume", resume)

    await service.tick(db_session, run.id)

    assert approved.proposal_action.id in resumed
    assert await _actions(db_session, run.id, "record_warning") == []


# --------------------------------------------------------------------------- carry-in 3: collateral


@pytest.mark.asyncio
async def test_collateral_roster_change_gives_the_affected_active_run_one_recovery_trigger(
    sessions, tmp_path, monkeypatch,  # noqa: F811
):
    from tests.test_orchestration_recovery_configuration import (
        apply_flow, approved_flow, arm_semantic_guard, assign, complete_baseline_processes, seed_goal_run, seed_spr180,
    )
    from huddleroom.schemas.orchestration_recovery import ToolReceiptV1

    seeded = await seed_spr180(sessions, tmp_path)
    async with sessions() as s:
        project = await s.get(Project, seeded.project_id)
        implementer = await s.get(Agent, seeded.implementer_id)
        other_goal, other_run = await seed_goal_run(s, project, objective="Implement the billing API")
        other_goal.constraints = {"deadline": "next sprint"}
        await assign(s, project.id, other_run.id, implementer, "implementation")
        await complete_baseline_processes(s, other_goal, other_run)
        await s.commit()
        other_run_id = other_run.id
    flow = await approved_flow(sessions, seeded)
    arm_semantic_guard(monkeypatch)

    receipt = await apply_flow(sessions, flow)

    assert isinstance(receipt, ToolReceiptV1) and receipt.status == "applied", receipt
    async with sessions() as s:
        triggers = await _actions(s, other_run_id, "recovery_trigger")
        assert len(triggers) == 1
        assert triggers[0].request["goal_id"] == str(other_goal.id)
        # The origin run's own lineage is not re-triggered by its own approved fix.
        origin = [t for t in await _actions(s, seeded.run_id, "recovery_trigger") if t.status == "reserved"]
        assert origin == []


# --------------------------------------------------------------------------- carry-in 5: seal-time validation


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["manager_mismatch", "unapproved_assignment"])
async def test_seal_time_validation_rejects_a_doomed_configuration_before_any_card(
    sessions, tmp_path, monkeypatch, case,  # noqa: F811
):
    from tests.test_orchestration_recovery_configuration import (
        arm_semantic_guard, invalid_arguments, propose, recovery_cards, seed_spr180,
    )
    from huddleroom.schemas.orchestration_recovery import TeamConfigurationApplyV1
    from huddleroom.services.orchestration_recovery_configuration import RecoveryConfigurationService

    seeded = await seed_spr180(sessions, tmp_path)
    arm_semantic_guard(monkeypatch)
    arguments = invalid_arguments(seeded, case)
    async with sessions() as s:
        goal = await s.get(OrchestrationGoal, seeded.goal_id)
        run = await s.get(OrchestrationRun, seeded.run_id)
        with pytest.raises(ValueError, match=case if case == "manager_mismatch" else None):
            await RecoveryConfigurationService().prepare(s, goal, run, TeamConfigurationApplyV1.model_validate(arguments))

    try:
        await propose(sessions, seeded, arguments)
    except Exception:  # noqa: BLE001 -- rejected while sealing
        pass
    assert await recovery_cards(sessions, seeded.goal_id) == []


# --------------------------------------------------------------------------- failed-gate replacement


@pytest.mark.asyncio
async def test_invalid_final_summary_creates_trigger_not_replacement_task(
    db_session, test_project, tmp_path, monkeypatch, stub_decision, analyzer_calls,
):
    from huddleroom.services.event_bus import emit_event_once
    from tests.conftest import heal_baseline_drift_for_test
    from tests.test_orchestration_final_summary_request import _request_summary

    # Same harness as tests/test_orchestration_final_summary_request.py (manual start, runnable workspace).
    async def _no_autostart(self, db, goal, run):
        return 0

    monkeypatch.setattr(OrchestrationService, "_start_released_tasks", _no_autostart)
    stub_decision(lambda _context: {"action_type": "noop"})
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    test_project.workspace_path = str(workspace.resolve())
    await db_session.flush()

    service, goal, run, writer, task, gate = await _request_summary(db_session, test_project)
    goal.weight = "trivial"
    await heal_baseline_drift_for_test(db_session, goal, run)
    invalid_session = Session(agent_id=writer.id, task_id=task.id, project_id=test_project.id, adapter_type="api",
                              status="completed", output="not json", metadata_={}, origin="auto")
    db_session.add(invalid_session)
    await db_session.flush()
    await emit_event_once(db_session, test_project.id, "session.completed",
                          {"session_id": str(invalid_session.id), "task_id": str(task.id)},
                          dedup_key=f"task12-invalid-summary:{invalid_session.id}")
    tasks_before = await db_session.scalar(select(func.count(Task.id)))

    await service.tick(db_session, run.id)
    await service.tick(db_session, run.id)

    failed_evidence = (await db_session.scalars(select(OrchestrationEvidence).where(
        OrchestrationEvidence.gate_id == gate.id, OrchestrationEvidence.source_id == invalid_session.id))).one()
    replacement_key = f"run:{run.id}:kind:create_delegation_task:final_summary:replacement:{failed_evidence.id}"
    assert (await db_session.scalars(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id, OrchestrationAction.idempotency_key == replacement_key))).all() == []
    assert await db_session.scalar(select(func.count(Task.id))) == tasks_before
    gate_triggers = [t for t in await _actions(db_session, run.id, "recovery_trigger")
                     if t.request["source_kind"] not in {"eventless_noop", "reported_blocker"}]
    assert len(gate_triggers) == 1


# --------------------------------------------------------------------------- follow-up: end-to-end lineages


def _retry_draft(task_id) -> "ProposalDraftV1":
    from huddleroom.schemas.orchestration_recovery import ProposalDraftV1

    return ProposalDraftV1.model_validate({
        "kind": "proposal", "diagnosis": "The task failed once; one approved retry should complete it.",
        "evidence": [{"label": "The task failed.", "value": {"status": "failed"}, "source_kind": "task",
                      "source_id": str(task_id)}],
        "problem": "The implementation task failed.", "general_updates": ["retry task"],
        "expected_effect": "The task runs again.",
        "calls": [{"tool": "orchestration.retry_task@1", "arguments": {"task_id": str(task_id)},
                   "rationale": "Retry the failed attempt."}],
    })


class _Scripted:
    def __init__(self, draft):
        self.draft = draft

    async def analyze(self, snapshot, *, project_id, evidence_query, on_raw=None):
        return self.draft


async def _approved_task_lineage(db, project, user, monkeypatch):
    """failed task -> recover_run trigger -> proposal -> approve_fix -> resume (apply + verify + retry wrapper)."""
    from tests.test_orchestration_basic_recovery import (
        _add_failed_session, _agent, _fake_task_run, _make_gate, _make_orchestrated_task, _make_run,
    )

    monkeypatch.setattr("huddleroom.services.orchestration_service.TaskService.run",
                        _fake_task_run([], mark_in_progress=True))
    developer = _agent("developer", "developer", ["implementation"])
    db.add(developer)
    await db.flush()
    service, goal, run = await _make_run(db, project.id)
    run.phase = "authorized"
    gate = await _make_gate(db, run.id)
    task = await _make_orchestrated_task(db, project.id, run.id, gate.id, developer.id)
    task.metadata_ = {**task.metadata_, "orchestration": {**task.metadata_["orchestration"], "goal_id": str(goal.id)}}
    await _add_failed_session(db, task, developer.id)
    await service.recover_run(db, run.id, baseline_ready=True)
    (trigger,) = await _actions(db, run.id, "recovery_trigger")
    svc = OrchestrationBlockerRecoveryService(analyzer=_Scripted(_retry_draft(task.id)))
    proposal = await svc.diagnose(db, trigger.id)
    decision = (await db.scalars(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.related_action_id == proposal.id))).one()
    await svc.reserve_answer(db, decision, selected_option="approve_fix", reason=None, user_id=user.id)
    status = await svc.resume(db, proposal.id)
    return SimpleNamespace(service=service, svc=svc, goal=goal, run=run, task=task, proposal=proposal,
                           trigger=trigger, status=status)


@pytest.mark.asyncio
async def test_task_failed_lineage_runs_approved_retry_and_closes(db_session, test_project, test_user, monkeypatch):
    """I3: trigger -> proposal -> approve_fix -> the approved retry really runs -> lineage resolves on progress."""
    flow = await _approved_task_lineage(db_session, test_project, test_user, monkeypatch)

    await db_session.refresh(flow.task)
    effects = [a for a in await _actions(db_session, flow.run.id, "retry_task") if a.idempotency_key.endswith(":effect")]
    assert len(effects) == 1 and effects[0].status == "completed" and flow.task.status == "in_progress"
    wrappers = [(a.action_type, a.status, a.error) for t in ("recovery_apply", "recovery_verify", "recovery_retry")
                for a in await _actions(db_session, flow.run.id, t)]
    await db_session.refresh(flow.proposal)
    assert flow.status == "retrying", (wrappers, flow.proposal.error)
    # The decision loop then does real work: the sealed decision_loop retry observes it and the lineage resolves.
    decision = await _decision(db_session, flow.run, {"action_type": "request_verification"})
    db_session.add(OrchestrationAction(run_id=flow.run.id, idempotency_key=f"progress:{uuid.uuid4()}",
                                       action_type="request_verification", request={}, status="completed",
                                       decision_id=decision.id))
    await db_session.flush()
    assert await flow.svc.resume(db_session, flow.proposal.id) == "resolved"
    warnings = (await db_session.scalars(select(OrchestrationWarning).where(
        OrchestrationWarning.run_id == flow.run.id, OrchestrationWarning.active.is_(True),
        OrchestrationWarning.warning_type == "recovery_blocked"))).all()
    assert warnings == []


@pytest.mark.asyncio
async def test_complete_retry_on_observed_progress_resolves_the_lineage(db_session, test_project, test_user, monkeypatch):
    """Fix round 4 (I1): an owner that observed progress outside the decision loop completes the reserved
    decision_loop retry; the proposal resolves and its blocker warning closes. Idempotent."""
    flow = await _approved_task_lineage(db_session, test_project, test_user, monkeypatch)
    assert flow.status == "retrying"

    assert await flow.svc.complete_retry_on_observed_progress(db_session, flow.proposal) is True
    assert await flow.svc.complete_retry_on_observed_progress(db_session, flow.proposal) is False

    assert await flow.svc.resume(db_session, flow.proposal.id) == "resolved"
    retry = (await _actions(db_session, flow.run.id, "recovery_retry"))[0]
    assert retry.status == "completed"
    warnings = (await db_session.scalars(select(OrchestrationWarning).where(
        OrchestrationWarning.run_id == flow.run.id, OrchestrationWarning.active.is_(True),
        OrchestrationWarning.warning_type == "recovery_blocked"))).all()
    assert warnings == []


@pytest.mark.asyncio
async def test_decision_loop_retry_does_not_hold_the_decision_loop(db_session, test_project, test_user, monkeypatch):
    """Only an approved proposal whose sealed retry is NOT the decision loop holds the loop while it
    is applying/verifying/retrying; a decision_loop retry needs the loop to run to observe progress."""
    flow = await _approved_task_lineage(db_session, test_project, test_user, monkeypatch)
    assert flow.status == "retrying" and flow.proposal.request["retry"]["target"]["kind"] == "decision_loop"
    assert await flow.service.recovery_holds_decision_loop(db_session, flow.run.id) is False

    # Contrast: the same retrying proposal sealed to a planning retry owns progress and holds the loop.
    flow.proposal.request = {**flow.proposal.request, "retry": {**flow.proposal.request["retry"],
                                                                "target": {"kind": "planning"}}}
    await db_session.flush()
    assert await flow.service.recovery_holds_decision_loop(db_session, flow.run.id) is True


@pytest.mark.asyncio
async def test_approved_retry_failing_again_stays_in_its_lineage(db_session, test_project, test_user, monkeypatch):
    """I4: the approved retry's attempt fails again -> the lineage re-diagnoses; recover_run starts no second one."""
    flow = await _approved_task_lineage(db_session, test_project, test_user, monkeypatch)
    effect = next(a for a in await _actions(db_session, flow.run.id, "retry_task") if a.idempotency_key.endswith(":effect"))
    retried = await db_session.get(Session, effect.target_id)
    retried.status, retried.metadata_ = "failed", {"attempt": {"effect_state": "started"}}
    flow.task.status = "failed"
    await db_session.flush()

    await flow.service.recover_run(db_session, flow.run.id, baseline_ready=True)
    await flow.service.recover_run(db_session, flow.run.id, baseline_ready=True)

    await db_session.refresh(flow.proposal)
    assert flow.proposal.status == "failed"
    triggers = await _actions(db_session, flow.run.id, "recovery_trigger")
    lineages = {(t.dispatch_contract or {}).get("root_trigger_action_id") or str(t.id) for t in triggers}
    assert lineages == {str(flow.trigger.id)} and len(triggers) == 2  # the root plus its attempt-2 follow-up
    assert [a for a in await _actions(db_session, flow.run.id, "failure_observed")
            if a.request.get("session_id") == str(retried.id)] == []
    assert await _actions(db_session, flow.run.id, "ask_human") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("fit_at_seal", [True, False])
async def test_invalid_final_summary_lineage_approves_gate_linked_replacement_and_run_completes(
    db_session, test_project, test_user, tmp_path, monkeypatch, stub_decision, fit_at_seal,
):
    """C1/I3: gate trigger -> proposal -> approve_fix -> the sealed retry re-runs the gate-linked replacement under
    the recovery effect key -> a valid replacement summary satisfies the gate and the run completes."""
    from huddleroom.schemas.orchestration_recovery import ProposalDraftV1
    from huddleroom.services.event_bus import emit_event_once
    from tests.conftest import heal_baseline_drift_for_test
    from tests.test_orchestration_final_summary_request import (
        _complete_goal_definition, _request_summary, _valid_summary_output,
    )
    from tests.test_orchestration_recovery_execution import Executors

    async def _no_autostart(self, db, goal, run):
        return 0

    monkeypatch.setattr(OrchestrationService, "_start_released_tasks", _no_autostart)
    stub_decision(lambda _context: {"action_type": "noop"})
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    test_project.workspace_path = str(workspace.resolve())
    await db_session.flush()
    service, goal, run, writer, task, gate = await _request_summary(db_session, test_project)
    goal.weight = "trivial"
    await heal_baseline_drift_for_test(db_session, goal, run)
    draft = ProposalDraftV1.model_validate({
        "kind": "proposal", "diagnosis": "The final summary was invalid; re-verify, then replace it once.",
        "evidence": [{"label": "The final summary gate failed.", "value": {"status": "failed"}, "source_kind": "gate",
                      "source_id": str(gate.id)}],
        "problem": "The final summary is invalid.", "general_updates": ["request verification"],
        "expected_effect": "A replacement summary is requested.",
        "calls": [{"tool": "orchestration.request_verification@1",
                   "arguments": {"gate_id": str(gate.id), "work_function": "validation"},
                   "rationale": "Record a fresh verification request."}],
    })

    async def analyze(self, snapshot, *, project_id, evidence_query, on_raw=None):
        return draft

    monkeypatch.setattr(RecoveryAnalyzer, "analyze", analyze)
    Executors(monkeypatch, SimpleNamespace(task=task), only={"request_verification"})
    invalid = Session(agent_id=writer.id, task_id=task.id, project_id=test_project.id, adapter_type="api",
                      status="completed", output="not json", metadata_={}, origin="auto")
    db_session.add(invalid)
    await db_session.flush()
    await emit_event_once(db_session, test_project.id, "session.completed",
                          {"session_id": str(invalid.id), "task_id": str(task.id)}, dedup_key=f"c1:{invalid.id}")

    if not fit_at_seal:  # N2: no strong summarization fit while the trigger is sealed
        writer.is_active = False
        await db_session.flush()
    await service.tick(db_session, run.id)  # gate trigger -> diagnosis -> pending fix card
    (proposal,) = await _actions(db_session, run.id, "recovery_proposal")
    assert proposal.request["retry"]["target"]["kind"] == "action"
    (marker,) = await _actions(db_session, run.id, "failure_observed")
    assert marker.request["retry_action"] == {"kind": "final_summary"}  # typed marker; built at retry time
    writer.is_active = True
    await db_session.flush()
    card = (await db_session.scalars(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.related_action_id == proposal.id))).one()
    await OrchestrationBlockerRecoveryService().reserve_answer(
        db_session, card, selected_option="approve_fix", reason=None, user_id=test_user.id)
    await _complete_goal_definition(db_session, goal, run)
    await service.tick(db_session, run.id)  # resume: apply -> verify -> the one sealed retry

    replacements = [a for a in await _actions(db_session, run.id, "create_delegation_task")
                    if a.idempotency_key.endswith(":retry:1:effect")]
    assert len(replacements) == 1 and replacements[0].status == "completed", [r.error for r in replacements]
    replacement = await db_session.get(Task, replacements[0].target_id)
    assert replacement.metadata_["orchestration"]["gate_id"] == str(gate.id)
    assert replacement.metadata_["orchestration"]["final_summary"] is True
    await db_session.refresh(proposal)
    assert proposal.status == "completed"  # resolved: verified and the one retry completed

    valid = Session(agent_id=replacement.assigned_to, task_id=replacement.id, project_id=test_project.id,
                    adapter_type="api", status="completed", output=await _valid_summary_output(db_session, goal, run),
                    metadata_={}, origin="auto")
    db_session.add(valid)
    await db_session.flush()
    await emit_event_once(db_session, test_project.id, "session.completed",
                          {"session_id": str(valid.id), "task_id": str(replacement.id)}, dedup_key=f"c1:{valid.id}")
    completed = await service.tick(db_session, run.id)

    await db_session.refresh(gate)
    assert gate.status == "accepted"
    assert completed["run_completed"] is True


@pytest.mark.asyncio
async def test_repeated_eventless_noops_and_reports_keep_one_open_lineage_each(db_session, test_project, analyzer_calls):
    """I2: a stuck loop's new noop / recover_blocker decisions join the open lineage instead of stacking new ones."""
    from tests.test_orchestration_runtime_e2e import _authorized_run

    service, goal, run = await _authorized_run(db_session, test_project)
    for _ in range(3):
        context = await service._decision_context(db_session, goal, run)
        noop = await _decision(db_session, run, {
            "action_type": "noop", "reason": "Waiting.",
            "wake_when": {"events": [], "recheck_after_seconds": 300, "expected_result": "Something changes."},
        }, snapshot=context)
        await service._dispatch_execution_decision(db_session, run, noop)
        report = await _decision(db_session, run, {
            "action_type": "recover_blocker", "reason": f"Still blocked {uuid.uuid4()}.",
            "evidence_refs": [{"kind": "decision", "id": str(noop.id)}],
        })
        await OrchestrationDecisionDispatcher(service).dispatch(db_session, run, report)

    triggers = await _actions(db_session, run.id, "recovery_trigger")
    assert sorted(t.request["source_kind"] for t in triggers) == ["eventless_noop", "reported_blocker"]
    reports = await _actions(db_session, run.id, "recover_blocker")
    assert len(reports) == 3 and {r.target_id for r in reports} == {
        next(t.id for t in triggers if t.request["source_kind"] == "reported_blocker")}


@pytest.mark.asyncio
async def test_tick_diagnoses_at_most_one_trigger_per_run(db_session, test_project, analyzer_calls, stub_decision):
    """M2: diagnosis is bounded to one analyzer exchange per run per tick."""
    from tests.test_orchestration_runtime_e2e import _authorized_run

    stub_decision(lambda _ctx: {"action_type": "ask_human", "question": "Proceed?", "reason": "Unsure."})
    service, goal, run = await _authorized_run(db_session, test_project)
    svc = OrchestrationBlockerRecoveryService()
    for code in ("first_blocker", "second_blocker"):
        await svc.create_trigger(db_session, _trigger(goal.id, run.id, blocker_code=code).model_copy(
            update={"source_fingerprint": (code[0] * 64)}))

    await service.tick(db_session, run.id)

    assert len(analyzer_calls) == 1
    assert len([t for t in await _actions(db_session, run.id, "recovery_trigger") if t.status == "reserved"]) == 1


@pytest.mark.asyncio
async def test_final_summary_retry_with_unavailable_manifest_fails_into_its_lineage(
    db_session, test_project, test_user, tmp_path, monkeypatch, stub_decision,
):
    """N3: a non-summary gate that is not accepted never aborts the failed-gate pass, and the approved retry then
    fails into the lineage as evidence (no replacement, no crash)."""
    from huddleroom.schemas.orchestration_recovery import ProposalDraftV1
    from huddleroom.services.event_bus import emit_event_once
    from tests.conftest import heal_baseline_drift_for_test
    from tests.test_orchestration_final_summary_request import _complete_goal_definition, _request_summary
    from tests.test_orchestration_recovery_execution import Executors

    async def _no_autostart(self, db, goal, run):
        return 0

    monkeypatch.setattr(OrchestrationService, "_start_released_tasks", _no_autostart)
    stub_decision(lambda _context: {"action_type": "noop"})
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    test_project.workspace_path = str(workspace.resolve())
    await db_session.flush()
    service, goal, run, writer, task, gate = await _request_summary(db_session, test_project)
    goal.weight = "trivial"
    await heal_baseline_drift_for_test(db_session, goal, run)
    draft = ProposalDraftV1.model_validate({
        "kind": "proposal", "diagnosis": "Replace the invalid final summary.",
        "evidence": [{"label": "The final summary gate failed.", "value": {"status": "failed"}, "source_kind": "gate",
                      "source_id": str(gate.id)}],
        "problem": "The final summary is invalid.", "general_updates": ["request verification"],
        "expected_effect": "A replacement summary is requested.",
        "calls": [{"tool": "orchestration.request_verification@1",
                   "arguments": {"gate_id": str(gate.id), "work_function": "validation"},
                   "rationale": "Record a fresh verification request."}],
    })

    async def analyze(self, snapshot, *, project_id, evidence_query, on_raw=None):
        return draft

    monkeypatch.setattr(RecoveryAnalyzer, "analyze", analyze)
    Executors(monkeypatch, SimpleNamespace(task=task), only={"request_verification"})

    async def manifest_unavailable(self, db, run_id):
        raise HTTPException(status_code=409, detail="All non-summary gates must be accepted and have accepted evidence")

    # Unavailable from the start: the failed-gate pass must not need it (seal is a typed marker only).
    monkeypatch.setattr(OrchestrationService, "_accepted_non_summary_manifest", manifest_unavailable)
    invalid = Session(agent_id=writer.id, task_id=task.id, project_id=test_project.id, adapter_type="api",
                      status="completed", output="not json", metadata_={}, origin="auto")
    db_session.add(invalid)
    await db_session.flush()
    await emit_event_once(db_session, test_project.id, "session.completed",
                          {"session_id": str(invalid.id), "task_id": str(task.id)}, dedup_key=f"n3:{invalid.id}")
    await service.tick(db_session, run.id)
    (proposal,) = await _actions(db_session, run.id, "recovery_proposal")
    card = (await db_session.scalars(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.related_action_id == proposal.id))).one()
    await OrchestrationBlockerRecoveryService().reserve_answer(
        db_session, card, selected_option="approve_fix", reason=None, user_id=test_user.id)
    await _complete_goal_definition(db_session, goal, run)

    await service.tick(db_session, run.id)  # must not raise

    await db_session.refresh(proposal)
    assert proposal.status == "failed"
    assert [a for a in await _actions(db_session, run.id, "create_delegation_task")
            if a.idempotency_key.endswith(":retry:1:effect")] == []
    follow_ups = [t for t in await _actions(db_session, run.id, "recovery_trigger")
                  if (t.dispatch_contract or {}).get("supersedes_proposal_action_id") == str(proposal.id)]
    assert len(follow_ups) == 1
    assert any("final_summary_unavailable" in str(item.get("value")) for item in follow_ups[0].dispatch_contract["evidence"])


@pytest.mark.asyncio
async def test_later_ordinary_failure_is_not_swallowed_by_an_earlier_recovery_attempt(
    db_session, test_project, test_user, monkeypatch,
):
    """N4: only the task's LATEST failed session is matched to a recovery effect; a later ordinary failure is
    handled on its own (here: the repeated-failure owner escalation), not absorbed into the old lineage."""
    flow = await _approved_task_lineage(db_session, test_project, test_user, monkeypatch)
    effect = next(a for a in await _actions(db_session, flow.run.id, "retry_task") if a.idempotency_key.endswith(":effect"))
    retried = await db_session.get(Session, effect.target_id)
    retried.status, retried.metadata_ = "failed", {"attempt": {"effect_state": "started"}}
    later = Session(agent_id=flow.task.assigned_to, task_id=flow.task.id, project_id=test_project.id,
                    adapter_type="api", status="failed", metadata_={"attempt": {"effect_state": "started"}},
                    origin="manual", created_at=datetime.now(timezone.utc) + timedelta(minutes=5))
    db_session.add(later)
    flow.task.status = "failed"
    await db_session.flush()

    assert await flow.service._recovery_effect_for_attempt(db_session, flow.run, flow.task) is None
    await flow.service.recover_run(db_session, flow.run.id, baseline_ready=True)

    await db_session.refresh(flow.proposal)
    assert flow.proposal.status == "reserved"  # the old lineage was not failed by the later ordinary failure
    assert len(await _actions(db_session, flow.run.id, "ask_human")) == 1


@pytest.mark.asyncio
async def test_a_trigger_whose_diagnosis_keeps_raising_does_not_starve_the_next(
    db_session, test_project, analyzer_calls, stub_decision, monkeypatch,
):
    """F: the per-tick diagnosis goes to the least recently attempted trigger, so a broken one rotates out."""
    from huddleroom.services.orchestration_recovery_evidence import RecoveryContextBuilder
    from tests.test_orchestration_runtime_e2e import _authorized_run

    stub_decision(lambda _ctx: {"action_type": "ask_human", "question": "Proceed?", "reason": "Unsure."})
    service, goal, run = await _authorized_run(db_session, test_project)
    svc = OrchestrationBlockerRecoveryService()
    broken = await svc.create_trigger(db_session, _trigger(goal.id, run.id, blocker_code="broken_blocker").model_copy(
        update={"source_fingerprint": "b" * 64}))
    healthy = await svc.create_trigger(db_session, _trigger(goal.id, run.id, blocker_code="healthy_blocker").model_copy(
        update={"source_fingerprint": "c" * 64}))
    original_build = RecoveryContextBuilder.build

    async def build(self, db, trigger, *, registry_manifest):
        if trigger.blocker_code == "broken_blocker":
            raise RuntimeError("context unavailable")
        return await original_build(self, db, trigger, registry_manifest=registry_manifest)

    monkeypatch.setattr(RecoveryContextBuilder, "build", build)

    await service.tick(db_session, run.id)  # the older, broken trigger is attempted first and fails
    await service.tick(db_session, run.id)  # it rotates out: the healthy one is diagnosed

    await db_session.refresh(broken)
    await db_session.refresh(healthy)
    assert broken.status == "reserved" and broken.dispatch_contract.get("last_diagnosis_error_at")
    assert healthy.status == "completed" and len(analyzer_calls) == 1


@pytest.mark.asyncio
async def test_analyzer_outage_backs_off_instead_of_calling_the_provider_every_tick(
    db_session, test_project, stub_decision, monkeypatch,
):
    """F3: an analyzer exception stamps the trigger; the next tick inside the backoff window makes no call."""
    from tests.test_orchestration_runtime_e2e import _authorized_run

    calls = []

    async def analyze(self, snapshot, *, project_id, evidence_query, on_raw=None):
        calls.append(project_id)
        raise RuntimeError("provider down")

    monkeypatch.setattr(RecoveryAnalyzer, "analyze", analyze)
    stub_decision(lambda _ctx: {"action_type": "ask_human", "question": "Proceed?", "reason": "Unsure."})
    service, goal, run = await _authorized_run(db_session, test_project)
    trigger = await OrchestrationBlockerRecoveryService().create_trigger(db_session, _trigger(goal.id, run.id))

    await service.tick(db_session, run.id)
    await service.tick(db_session, run.id)

    await db_session.refresh(trigger)
    assert len(calls) == 1
    assert trigger.status == "reserved" and trigger.dispatch_contract.get("last_diagnosis_error_at")
