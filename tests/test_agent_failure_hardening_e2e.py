"""One goal run in which the agents misbehave the way real runs do, driven through real ticks, delegation, session
completion and evidence ingestion. Only the model boundary is faked: the orchestrator's decisions, the agents'
outputs and the recovery analyzer (which is unavailable throughout)."""
import json
import uuid

import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def test_engine(tmp_path):
    """SPR230/SPR231 commit-capable fixtures stay isolated per test."""
    from sqlalchemy import event
    from sqlalchemy.ext.asyncio import create_async_engine
    from huddleroom.models.base import Base
    from tests.conftest import _seed_anon_actors

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'owned-setup.db'}")
    @event.listens_for(engine.sync_engine, "connect")
    def pragmas(connection, _record):
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        await _seed_anon_actors(connection)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture(autouse=True)
async def clean_test_database():
    # The database file and engine are unique to this test; disposal replaces global row deletion.
    yield


@pytest_asyncio.fixture
async def db_session(test_engine, monkeypatch):
    from sqlalchemy.ext.asyncio import AsyncSession

    monkeypatch.setattr("huddleroom.workers.task_runner.register_session", lambda *_args, **_kwargs: None)
    async with AsyncSession(test_engine, expire_on_commit=False) as db:
        yield db
        await db.rollback()

from sqlalchemy import select

from tests.conftest import adopt_session
from huddleroom.config import settings
from huddleroom.models.artifact import Artifact
from huddleroom.models.orchestration import (
    OrchestrationAction,
    OrchestrationDecision,
    OrchestrationEvidence,
    OrchestrationGate,
    OrchestrationWait,
)
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision, OrchestrationWarning
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.services import orchestration_review_brief as rb
from huddleroom.services.orchestration_decision_dispatcher import OrchestrationDecisionDispatcher
from huddleroom.services.orchestration_plan_ingestion import plan_artifact_id
from huddleroom.services.orchestration_progress_view import OrchestrationProgressView
from huddleroom.services.orchestration_recovery_analyzer import RecoveryAnalyzer
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.orchestration_work_report import CANONICAL_WORK_REPORT_SCHEMA
from tests.test_orchestration_final_summary_request import runnable_workspace  # noqa: F401  (autouse)
from tests.test_orchestration_runtime_e2e import _agent, _authorized_run, _complete_task_session, _tasks_for_run


@pytest.fixture(autouse=True)
def _manual_task_start(monkeypatch):
    async def _noop(self, db, goal, run):
        return 0
    monkeypatch.setattr(OrchestrationService, "_start_released_tasks", _noop)


CRITERIA = [
    {"key": "checklist", "description": "A product checklist is agreed for the milestone."},
    {"key": "research", "description": "Commercial feasibility is established from dated public sources."},
]
REPORT = {
    "status": "completed", "changes": [], "evidence": [], "criterion_progress": {}, "decisions": [], "risks": [],
    "open_questions": [], "next_step": "Request verification.", "collaboration_need": None,
}
RESEARCH_EVIDENCE = "Provider price list dated 2026-10-01: 0.004 USD per message; 5,000 messages cost 20 USD."
OWNER_QUESTIONS = (
    "What monthly messaging budget and commercial permission apply to this account?",
    "Can you share the redacted notes of the earlier one-contact smoke test, or confirm none were kept?",
)


def _envelope(kind, research_keys, product_id, researcher_id):
    return json.dumps({
        "schema_version": 1, "plan_kind": kind, "summary": "A checklist and a feasibility study.",
        "plan_items": [
            {"id": "checklist", "work_function": "planning", "scope": "Draft the product checklist.",
             "deliverable": "Product checklist.", "why_runnable": "No dependencies.",
             "success_criterion_keys": ["checklist"], "agent_id": product_id},
            {"id": "research", "work_function": "investigation", "scope": "Research commercial feasibility.",
             "deliverable": "Dated, sourced cost calculations.", "why_runnable": "No dependencies.",
             "success_evidence": ["Source-dated price table."], "success_criterion_keys": research_keys,
             "agent_id": researcher_id},
        ],
    })


def _schema_in_prompt(description):
    tail = description.split("Read-only work reports its findings in this JSON.\n", 1)[1]
    return json.JSONDecoder().raw_decode(tail)[0]


def _orchestrator(ctx):
    """A sensible orchestrator: accept an ingested plan, verify finished work, otherwise wait for agent work."""
    state = ctx["run"]["plan_state"]
    if state.get("plan_artifact_id") and state.get("status") != "accepted":
        return {"action_type": "accept_plan", "plan_artifact_id": state["plan_artifact_id"]}
    work_gates = {gate["id"] for gate in ctx["gates"] if gate.get("gate_type") == "work_completed"}
    for follow_up in ctx["untracked_follow_ups"]:
        if follow_up["kind"] == "unverified_gate" and follow_up["id"] in work_gates:
            return {"action_type": "request_verification", "gate_id": follow_up["id"], "work_function": "validation"}
    active = [task["id"] for task in ctx["tasks"] if task.get("status") in ("backlog", "ready", "in_progress")]
    return {"action_type": "noop", "reason": "Agent work is in progress.", "wake_when": {
        "events": [{"event_type": "task.status_changed", "matcher": {"task_id": task_id}} for task_id in active[:5]],
        "recheck_after_seconds": 300, "expected_result": "Agent work reports back.",
    }}


async def _tick(service, db, run, stub_decision, decision=_orchestrator):
    stub_decision(decision)
    await db.flush()
    await db.commit()  # This driver owns the preceding scenario setup, never an external caller transaction.
    result = await service.tick(db, run.id)
    await db.refresh(run)
    return result


async def _dispatch(db, service, run, parsed):
    """Apply one orchestrator decision through the dispatcher a tick uses."""
    decision = OrchestrationDecision(run_id=run.id, decision_type="coordination", input_snapshot={},
                                     validator_status="accepted", parsed_decision=parsed)
    db.add(decision)
    await db.flush()
    await db.commit()  # This scenario owns its prepared immutable decision before native dispatch.
    return await OrchestrationDecisionDispatcher(service).dispatch(db, run, decision)


async def _marker(service, db, run, task):
    return await service._existing_action_for_key(db, run.id, f"run:{run.id}:kind:report_consumed:task:{task.id}")


async def _children(db, task):
    """Output-only corrections of ``task``'s result."""
    rows = await db.scalars(select(Task).where(Task.parent_id == task.id))
    return [row for row in rows if (row.metadata_ or {}).get("result_correction_parent_task_id") == str(task.id)]


async def _verifications(db, gate):
    return list(await db.scalars(select(OrchestrationEvidence).where(
        OrchestrationEvidence.gate_id == gate.id, OrchestrationEvidence.source_type == "verification")))


def _verifies(task, gate):
    return task.metadata_["orchestration"].get("verification_gate_id") == str(gate.id)


async def _criteria(db, goal, run):
    view = await OrchestrationProgressView().build(db, goal, run)
    return {entry["criterion_key"]: entry["state"] for entry in view.progress_view}


@pytest.mark.usefixtures("safe_effectiveness_review_continue")
async def test_misbehaving_agents_are_corrected_without_owner_or_duplicate_work(
    db_session, test_project, stub_decision, monkeypatch, owned_session_factory,
):
    adopt_session(db_session)
    monkeypatch.setattr(settings, "orchestration_max_actions_per_tick", 1)
    analyzed = []

    async def unavailable_analyzer(self, snapshot, **_kw):
        analyzed.append(snapshot["source_error"])
        raise RuntimeError("recovery provider unavailable")

    monkeypatch.setattr(RecoveryAnalyzer, "analyze", unavailable_analyzer)
    planner, product = _agent("lead", ["planning"]), _agent("product", ["planning"])
    researcher, verifier = _agent("researcher", ["investigation"]), _agent("reviewer", ["validation"])
    researcher.role, verifier.role = "researcher", "validator"
    db_session.add_all([planner, product, researcher, verifier])
    await db_session.flush()
    service, goal, run = await _authorized_run(db_session, test_project, success_criteria=CRITERIA, owns_setup=True)

    # --- Planning: the lead is asked for a plan, which is late.
    await _tick(service, db_session, run, stub_decision, lambda _ctx: {
        "action_type": "request_plan", "agent_id": str(planner.id), "work_function": "planning",
        "scope": "Plan the milestone.",
    })
    planning_task = await db_session.get(Task, uuid.UUID(run.plan_state["planning_task_id"]))
    version = run.plan_state["planning_version"]
    kind = planning_task.metadata_["orchestration_contract"]["report_schema"]["plan_kind"]
    plan_missing = {"condition": "plan_missing", "subject": f"{run.id}:{version}"}
    await _tick(service, db_session, run, stub_decision, lambda _ctx: {
        "action_type": "record_warning", "warning_type": "plan_missing", "severity": "warning",
        "message": "The plan has not arrived.", "condition": plan_missing,
    })
    await _tick(service, db_session, run, stub_decision, lambda _ctx: {
        "action_type": "ask_human", "question": "Planning is slow; should it continue?", "condition": plan_missing,
    })
    [late_warning] = list(await db_session.scalars(select(OrchestrationWarning).where(
        OrchestrationWarning.run_id == run.id, OrchestrationWarning.warning_type == "plan_missing")))
    [late_question] = list(await db_session.scalars(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.run_id == run.id, OrchestrationAuthorityDecision.status == "pending")))

    # The lead answers with a framing sentence around an envelope citing an unknown criterion key.
    failed_plan = "Now writing the plan envelope.\n" + _envelope(
        kind, ["protocol_fallback"], str(product.id), str(researcher.id))
    planner_session = await _complete_task_session(
        db_session, test_project.id, planning_task, planner.id, failed_plan)
    await _tick(service, db_session, run, stub_decision)

    marker = await _marker(service, db_session, run, planning_task)
    assert marker.request["normalized"] == "framing"  # the framing alone was not a failure
    assert "protocol_fallback" in marker.request["plan_ingestion_error"]
    assert run.plan_state["status"] != "revision_required" and "plan_artifact_id" not in run.plan_state
    [plan_fix] = await _children(db_session, planning_task)
    assert plan_fix.assigned_to == planner.id
    assert failed_plan in plan_fix.description and marker.request["plan_ingestion_error"] in plan_fix.description
    await db_session.refresh(late_warning)
    await db_session.refresh(late_question)
    assert late_warning.active and late_question.status == "pending"

    # The output-only correction returns the fixed envelope: it lands under the original planning identity and
    # the orchestrator accepts it.
    fixed_plan = _envelope(kind, ["research"], str(product.id), str(researcher.id))
    plan_fix_session = await _complete_task_session(db_session, test_project.id, plan_fix, planner.id, fixed_plan)
    await _tick(service, db_session, run, stub_decision)

    expected_plan = plan_artifact_id(planning_task.id, version)
    assert run.plan_state["status"] == "accepted"
    assert run.plan_state["accepted_artifact_id"] == str(expected_plan)
    assert run.plan_state["planning_task_id"] == str(planning_task.id)
    marker = await _marker(service, db_session, run, planning_task)
    assert marker.request["result_correction"]["status"] == "corrected"
    assert marker.request["result_correction"]["correction_session_id"] == str(plan_fix_session.id)

    # --- Execution: the next tick releases both plan items and reconciles what the accepted plan resolved.
    await _tick(service, db_session, run, stub_decision)
    await db_session.refresh(late_warning)
    await db_session.refresh(late_question)
    assert not late_warning.active and late_warning.resolved_by == "system"
    assert late_question.status == "cancelled"
    work = [task for task in await _tasks_for_run(db_session, run.id)
            if (task.metadata_ or {}).get("orchestration", {}).get("plan_item_gate_id")]
    checklist = next(task for task in work if service._task_work_function(task) == "planning")
    research = next(task for task in work if service._task_work_function(task) == "investigation")
    assert (checklist.assigned_to, research.assigned_to) == (product.id, researcher.id)
    # The gated planning-role checklist is briefed with, and validated against, the WorkReport.
    assert checklist.metadata_["orchestration_contract"]["expected_result"] == "work_report"
    assert _schema_in_prompt(checklist.description) == CANONICAL_WORK_REPORT_SCHEMA
    assert "Planning rules:" not in checklist.description

    # Two unrelated blockers are reported while the work runs; a repeat of the first joins its lineage.
    reports = []
    for task, reason in ((checklist, "The checklist owner is unclear."), (research, "Research sources are paywalled."),
                         (checklist, "The checklist owner is still unclear.")):
        reports.append(await _dispatch(db_session, service, run, {
            "action_type": "recover_blocker", "reason": reason, "evidence_refs": [{"kind": "task", "id": str(task.id)}],
        }))
    assert reports[0].target_id == reports[2].target_id != reports[1].target_id

    checklist_report = {**REPORT, "changes": ["Drafted the product checklist."],
                        "evidence": ["Checklist reviewed against the milestone scope."],
                        "criterion_progress": {"checklist": "candidate"}}
    research_report = {**REPORT, "changes": ["Compared provider pricing from dated public sources."],
                       "evidence": [RESEARCH_EVIDENCE], "criterion_progress": {"research": "candidate"}}
    await _complete_task_session(db_session, test_project.id, checklist, product.id, json.dumps(checklist_report))
    # The research result exists only in the database: no workspace file, no artifact.
    await _complete_task_session(db_session, test_project.id, research, researcher.id, json.dumps(research_report))
    for _ in range(3):  # consume both reports, then the orchestrator verifies each gate
        await _tick(service, db_session, run, stub_decision)

    for task, report in ((checklist, checklist_report), (research, research_report)):
        consumed = await _marker(service, db_session, run, task)
        assert consumed.request["report"]["evidence"] == report["evidence"]  # valid the first time
        assert "result_correction" not in consumed.request and await _children(db_session, task) == []
    triggers = list(await db_session.scalars(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == "recovery_trigger")))
    assert sorted(trigger.request["summary"] for trigger in triggers) == [
        "Research sources are paywalled.", "The checklist owner is unclear."]
    assert len({trigger.request["source_fingerprint"] for trigger in triggers}) == 2
    # Each lineage got its own diagnosis attempt; the unavailable analyzer is backed off, not retried every tick.
    assert sorted((item["source_id"], item["summary"]) for item in analyzed) == sorted(
        (trigger.request["source_id"], trigger.request["summary"]) for trigger in triggers)

    checklist_gate = await db_session.get(OrchestrationGate, uuid.UUID(
        checklist.metadata_["orchestration"]["plan_item_gate_id"]))
    research_gate = await db_session.get(OrchestrationGate, uuid.UUID(
        research.metadata_["orchestration"]["plan_item_gate_id"]))
    verifier_tasks = [task for task in await _tasks_for_run(db_session, run.id)
                      if service._task_work_function(task) == "validation"]
    [checklist_check] = [t for t in verifier_tasks if _verifies(t, checklist_gate)]
    [research_check] = [t for t in verifier_tasks if _verifies(t, research_gate)]
    assert {checklist_check.assigned_to, research_check.assigned_to} == {verifier.id}
    # The verifier of the DB-only research receives the producer's report, its scope and the criterion text.
    brief = research_check.description
    fenced = brief.split("<<<PRODUCER_REPORT\n", 1)[1].split("\nPRODUCER_REPORT>>>", 1)[0]
    assert RESEARCH_EVIDENCE in fenced and rb.UNTRUSTED_REPORT_NOTE in brief
    assert "Research commercial feasibility." in brief and CRITERIA[1]["description"] in brief

    # --- Verification: the checklist verdict is well formed; the research verdict is prose.
    await _complete_task_session(db_session, test_project.id, checklist_check, verifier.id, json.dumps(
        {"status": "done", "verdict": "accepted", "evidence": ["The checklist covers every milestone item."]}))
    prose = "I read the pricing research and it looks right to me."
    prose_session = await _complete_task_session(db_session, test_project.id, research_check, verifier.id, prose)
    await _tick(service, db_session, run, stub_decision)

    await db_session.refresh(checklist_gate)
    await db_session.refresh(research_gate)
    assert checklist_gate.status == "accepted"
    assert research_gate.status == "open" and await _verifications(db_session, research_gate) == []
    assert await _criteria(db_session, goal, run) == {"checklist": "accepted", "research": "evidence_pending"}
    [verdict_fix] = await _children(db_session, research_check)
    assert verdict_fix.assigned_to == verifier.id and prose in verdict_fix.description

    verdict_fix_session = await _complete_task_session(
        db_session, test_project.id, verdict_fix, verifier.id, json.dumps(
            {"status": "done", "verdict": "accepted", "evidence": ["Prices match the dated provider list."]}))
    await _tick(service, db_session, run, stub_decision)

    await db_session.refresh(research_gate)
    assert research_gate.status == "accepted"
    [verification] = await _verifications(db_session, research_gate)
    assert verification.source_id == verdict_fix_session.id
    assert await _criteria(db_session, goal, run) == {"checklist": "accepted", "research": "accepted"}

    # --- The research raised two account questions only the owner can answer.
    for question in OWNER_QUESTIONS:
        await _dispatch(db_session, service, run, {"action_type": "ask_human", "question": question})
    final = await _tick(service, db_session, run, stub_decision)

    # One canonical plan artifact, the accepted one.
    plans = list(await db_session.scalars(select(Artifact.id).where(
        Artifact.project_id == test_project.id, Artifact.artifact_type == "plan")))
    assert plans == [expected_plan]
    # Raw failures are preserved beside their corrections.
    marker = await _marker(service, db_session, run, planning_task)
    assert marker.request["rejected_output"] == failed_plan and marker.request["session_id"] == str(planner_session.id)
    assert (await db_session.get(Session, planner_session.id)).output == failed_plan
    verdict_ledger = (await _marker(service, db_session, run, research_check)).request["result_correction"]
    assert verdict_ledger["status"] == "corrected"
    assert [attempt["session_id"] for attempt in verdict_ledger["attempts"]] == [str(prose_session.id)]
    assert (await db_session.get(Session, prose_session.id)).output == prose
    # No duplicated work: one task and one session per plan item and verification, one correction per failure.
    tasks = await _tasks_for_run(db_session, run.id)
    assert sorted(service._task_work_function(task) for task in tasks) == sorted(
        ["planning", "report_clarification", "planning", "investigation", "validation", "validation",
         "report_clarification"])
    for task in tasks:
        assert len(list(await db_session.scalars(select(Session.id).where(Session.task_id == task.id)))) == 1
    assert [task.id for task in await _children(db_session, planning_task)] == [plan_fix.id]
    assert [task.id for task in await _children(db_session, research_check)] == [verdict_fix.id]
    # The run waits only on the two genuine owner questions: no correction, review or recovery escalations.
    pending = list(await db_session.scalars(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.run_id == run.id, OrchestrationAuthorityDecision.status == "pending")))
    assert sorted(decision.question for decision in pending) == sorted(OWNER_QUESTIONS)
    waits = list(await db_session.scalars(select(OrchestrationWait).where(
        OrchestrationWait.run_id == run.id, OrchestrationWait.status == "open")))
    assert sorted(wait.owner["decision_id"] for wait in waits) == sorted(str(decision.id) for decision in pending)
    assert final["authorized_execution"]["outcome"] == "waiting"
    assert final["authorized_execution"]["wait_id"] in {str(wait.id) for wait in waits}
    assert not [blocker for blocker in run.active_blockers if blocker.get("kind") == "owner_escalation"]
