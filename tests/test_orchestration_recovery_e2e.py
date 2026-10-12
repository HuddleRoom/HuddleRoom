"""SPR 180/184: integrated acceptance of the activated blocker-recovery feature.

Flows are driven through the real ``OrchestrationService.tick`` and the HTTP decision answer endpoint where
practical, and through the service re-entry point (``resume``, which is exactly what the tick calls) where a
tick cannot isolate one phase. Fake completions only: the ordinary orchestrator decision is ``stub_decision``,
the recovery analyzer is a scripted stand-in, CLI executables are faked (``shutil.which``) and never launched,
and sessions are never dispatched.
"""
from __future__ import annotations

import uuid

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select

from huddleroom.models.agent import Agent
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGoal, OrchestrationRun, OrchestrationWait
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision, OrchestrationWarning
from huddleroom.models.project import Project
from huddleroom.models.task import Task
from huddleroom.models.user import User
from huddleroom.schemas.orchestration_recovery import CapabilityGapAssessmentV1
from huddleroom.services import orchestration_baseline_runtime as runtime
from huddleroom.services.orchestration_blocker_recovery import OrchestrationBlockerRecoveryService
from huddleroom.services.orchestration_process_service import OrchestrationProcessService
from huddleroom.services.orchestration_recovery_analyzer import RecoveryAnalyzer
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.orchestration_team_hierarchy import select_plan_author
from tests.conftest import complete_baseline_processes
from tests.test_orchestration_recovery_configuration import (
    PLANNER_DESCRIPTION, arm_semantic_guard, configuration_arguments, proposal_draft, seed_spr180,
    snapshot_configuration_state,
)
from tests.test_orchestration_roster_mutation import assign, seed_goal_run, sessions  # noqa: F401

pytestmark = pytest.mark.usefixtures(
    "safe_goal_analysis", "safe_agent_definition_review", "safe_effectiveness_review_continue",
)

FIX_OPTIONS = ["approve_fix", "reject"]


# --------------------------------------------------------------------------- shared helpers


def decisions_url(project_id, goal_id) -> str:
    return f"/api/v1/projects/{project_id}/orchestration/goals/{goal_id}/decisions"


def committed_client(test_engine, user_id):
    """HTTP client whose sessions behave like production get_db (commit on success, rollback on error)."""
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from huddleroom.database import get_db
    from huddleroom.dependencies import get_current_user
    from huddleroom.main import create_app

    app = create_app()
    factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)

    async def override_get_db():
        async with factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def current_user():
        async with factory() as session:
            return await session.get(User, user_id)

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = current_user
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def answer(client, project_id, goal_id, decision_id, headers=None, **payload):
    return await client.post(f"{decisions_url(project_id, goal_id)}/{decision_id}/answer", json=payload,
                             headers=headers or {})


async def projected(client, project_id, goal_id, decision_id, headers=None) -> dict:
    response = await client.get(decisions_url(project_id, goal_id), headers=headers or {})
    assert response.status_code == 200, response.text
    body = response.json()
    items = body["items"] if isinstance(body, dict) else body
    return next(item for item in items if item["id"] == str(decision_id))


async def actions(s, run_id, action_type):
    return list((await s.scalars(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run_id, OrchestrationAction.action_type == action_type,
    ).order_by(OrchestrationAction.created_at, OrchestrationAction.id)
        .execution_options(populate_existing=True))).all())


async def recovery_cards(s, goal_id, *, status=None):
    rows = (await s.scalars(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.goal_id == goal_id,
        OrchestrationAuthorityDecision.decision_key.startswith("recovery"),
    ).execution_options(populate_existing=True))).all()
    return [row for row in rows if status is None or row.status == status]


def fix_cards(cards):
    return [row for row in cards if [option["key"] if isinstance(option, dict) else option
                                     for option in (row.options or [])] == FIX_OPTIONS]


async def active_warnings(s, goal_id, warning_type=None):
    rows = (await s.scalars(select(OrchestrationWarning).where(
        OrchestrationWarning.goal_id == goal_id, OrchestrationWarning.active.is_(True),
    ).execution_options(populate_existing=True))).all()
    return [row for row in rows if warning_type is None or row.warning_type == warning_type]


def script_analyzer(monkeypatch, *outcomes):
    """Every RecoveryAnalyzer instance (the tick's too) returns the queued outcomes in order; the last repeats."""
    queue = list(outcomes)
    calls: list = []

    async def analyze(self, snapshot, *, project_id, evidence_query, on_raw=None):
        calls.append(snapshot)
        outcome = queue.pop(0) if len(queue) > 1 else queue[0]
        if callable(outcome):
            outcome = outcome()
        if on_raw is not None:
            on_raw(outcome.model_dump_json())
        return outcome

    monkeypatch.setattr(RecoveryAnalyzer, "analyze", analyze)
    return calls


def gap_assessment(source_id) -> CapabilityGapAssessmentV1:
    return CapabilityGapAssessmentV1.model_validate({
        "kind": "capability_gap",
        "diagnosis": "No registered tool can repair this blocker.",
        "evidence": [{"label": "The run is blocked.", "value": {"status": "blocked"}, "source_kind": "run",
                      "source_id": str(source_id)}],
        "missing_capability": "Access to the staging environment for the release check.",
        "required_human_action": "Grant the release check access to staging.",
        "wake_after": {"kind": "human_decision", "decision": "Staging access granted"},
    })


def no_session_dispatch(monkeypatch):
    from huddleroom.services.session_service import SessionService

    async def _no_autostart(self, db, goal, run):
        return 0

    monkeypatch.setattr(OrchestrationService, "_start_released_tasks", _no_autostart)
    monkeypatch.setattr(SessionService, "_schedule_dispatch_after_commit", lambda *args, **kwargs: None)


async def tick(sessions, run_id):  # noqa: F811
    async with sessions() as s:
        return await OrchestrationService().tick(s, run_id)


# --------------------------------------------------------------------------- SPR 180 harness (committed sessions)


async def started_spr180(sessions, tmp_path, monkeypatch, stub_decision, *, second_goal=False):  # noqa: F811
    """SPR 180 state (planning assigned to a planner whose accepted eligibility is empty), Started, with the
    decision loop asking that planner for a plan. ``second_goal`` adds another active goal sharing the roster."""
    seeded = await seed_spr180(sessions, tmp_path, mode="update")
    seeded.other_goal_id = seeded.other_run_id = None
    if second_goal:
        async with sessions() as s:
            project = await s.get(Project, seeded.project_id)
            implementer = await s.get(Agent, seeded.implementer_id)
            other_goal, other_run = await seed_goal_run(s, project, objective="Implement the billing API")
            other_goal.constraints = {"deadline": "next sprint"}
            await assign(s, project.id, other_run.id, implementer, "implementation")
            await complete_baseline_processes(s, other_goal, other_run)
            await s.commit()
            seeded.other_goal_id, seeded.other_run_id = other_goal.id, other_run.id
    no_session_dispatch(monkeypatch)
    stub_decision(lambda _ctx: {"action_type": "request_plan", "agent_id": str(seeded.planner_id),
                                "work_function": "planning", "scope": "Plan the release API."})
    async with sessions() as s:
        service = OrchestrationService()
        run = await s.get(OrchestrationRun, seeded.run_id)
        if run.phase != "ready":
            await service.tick(s, run.id)
            await s.refresh(run)
        assert run.phase == "ready", run.phase
        _goal, run = await service.start_run(s, seeded.project_id, seeded.goal_id, actor="human:test")
        assert run.phase == "authorized"
        await s.commit()
    seeded.analyzer_calls = script_analyzer(monkeypatch, proposal_draft(seeded, configuration_arguments(seeded)))
    return seeded


async def diagnosed_spr180(sessions, seeded):  # noqa: F811
    """One tick: request_plan -> plan_author_unavailable -> trigger -> committed -> diagnosis -> fix card."""
    await tick(sessions, seeded.run_id)
    async with sessions() as s:
        (proposal,) = await actions(s, seeded.run_id, "recovery_proposal")
        (card,) = await recovery_cards(s, seeded.goal_id)
        seeded.proposal_id, seeded.card_id = proposal.id, card.id
    return seeded


async def process_history(s, goal_id) -> dict:
    rows = (await OrchestrationProcessService().list_process_runs(s, goal_id))
    counts: dict = {}
    for row in rows:
        if row.process_type in {"agent_definition_review", "team_hierarchy"}:
            key = (row.process_type, row.trigger_reason)
            counts[key] = counts.get(key, 0) + 1
    return counts


async def plan_author(s, seeded):
    goal = await s.get(OrchestrationGoal, seeded.goal_id, populate_existing=True)
    run = await s.get(OrchestrationRun, seeded.run_id, populate_existing=True)
    try:
        return await select_plan_author(s, goal, run)
    except HTTPException:
        return None


async def pending_review_keys(s, goal_id):
    rows = (await s.scalars(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.goal_id == goal_id, OrchestrationAuthorityDecision.status == "pending",
    ).execution_options(populate_existing=True))).all()
    return [row.decision_key for row in rows
            if row.decision_key.startswith(("agent_definition_review:", "team_hierarchy:"))]


# --------------------------------------------------------------------------- SPR 180: direct configuration


@pytest.mark.asyncio
async def test_missing_plan_author_is_repaired_after_one_fix_approval(
    sessions, tmp_path, monkeypatch, stub_decision, test_engine,  # noqa: F811
):
    seeded = await started_spr180(sessions, tmp_path, monkeypatch, stub_decision)
    async with sessions() as s:
        assert await plan_author(s, seeded) is None, "precondition: SPR 180 state"
        history_before = await process_history(s, seeded.goal_id)

    await diagnosed_spr180(sessions, seeded)

    async with sessions() as s:
        (trigger,) = await actions(s, seeded.run_id, "recovery_trigger")
        assert trigger.request["source_kind"] == "planning_preflight"
        assert trigger.request["blocker_code"] == "plan_author_unavailable"
        assert trigger.request["retry_target"]["kind"] == "planning"
        assert len(fix_cards(await recovery_cards(s, seeded.goal_id, status="pending"))) == 1
    async with committed_client(test_engine, seeded.user_id) as client:
        card = await projected(client, seeded.project_id, seeded.goal_id, seeded.card_id)
        recovery = card["recovery"]
        assert recovery["status"] == "pending_approval"
        assert recovery["problem"] and recovery["after_approval"].startswith("HuddleRoom applies these updates")
        assert recovery["global_effect"] is True
        assert any("planning once" in bullet.lower() for bullet in recovery["general_updates"])
        (call,) = recovery["exact_details"]["calls"]
        assert call["tool"] == "team_configuration.apply"
        assert {change["target_type"] for change in call["changes"]} >= {"agent", "eligibility", "assignments"}

        first = await answer(client, seeded.project_id, seeded.goal_id, seeded.card_id, selected_option="approve_fix")
        duplicate = await answer(client, seeded.project_id, seeded.goal_id, seeded.card_id,
                                 selected_option="approve_fix")
        assert first.status_code == 200, first.text
        assert duplicate.status_code == 200, duplicate.text
        assert first.json()["decision"]["recovery"]["status"] == "applying"  # not resolved before the retry
        async with sessions() as s:
            assert len(await actions(s, seeded.run_id, "recovery_apply")) == 1
            assert await plan_author(s, seeded) is None, "approval alone changes no configuration"

        arm_semantic_guard(monkeypatch)  # the approved fix must never run a definition/hierarchy LM review
        await tick(sessions, seeded.run_id)  # resume: apply -> verify -> the one planning retry
        async with sessions() as s:
            after_first_tick = await snapshot_configuration_state(s, seeded.goal_id)
        await tick(sessions, seeded.run_id)  # replay
        await tick(sessions, seeded.run_id)

        resolved = await projected(client, seeded.project_id, seeded.goal_id, seeded.card_id)
        assert resolved["recovery"]["status"] == "resolved"

    async with sessions() as s:
        proposal = await s.get(OrchestrationAction, seeded.proposal_id, populate_existing=True)
        assert proposal.status == "completed"
        pid = proposal.request["proposal_id"]
        wrappers = {kind: [row for row in await actions(s, seeded.run_id, kind)
                           if row.idempotency_key.startswith(f"recovery:{pid}:")]
                    for kind in ("recovery_apply", "recovery_verify", "recovery_retry")}
        assert [len(rows) for rows in wrappers.values()] == [1, len(proposal.request["verification"]), 1]
        assert all(row.status == "completed" for rows in wrappers.values() for row in rows)
        planning_retries = [row for row in await actions(s, seeded.run_id, "request_plan")
                            if row.idempotency_key.endswith(":retry:1:effect")]
        planning_retry_count = len(planning_retries)
        assert planning_retry_count == 1 and planning_retries[0].status == "completed"
        # One approval atomically installed recovery-sourced snapshots: no review / hierarchy rerun.
        history = await process_history(s, seeded.goal_id)
        added = {key: count - history_before.get(key, 0) for key, count in history.items()
                 if count != history_before.get(key, 0)}
        assert added == {("agent_definition_review", "approved_recovery"): 1,
                         ("team_hierarchy", "approved_recovery"): 1}
        assert await pending_review_keys(s, seeded.goal_id) == []
        planner = await s.get(Agent, seeded.planner_id, populate_existing=True)
        assert planner.description == PLANNER_DESCRIPTION
        assert await plan_author(s, seeded) == seeded.planner_id
        run = await s.get(OrchestrationRun, seeded.run_id, populate_existing=True)
        assert run.plan_state["status"] == "requested"
        assert run.plan_state["plan_author_agent_id"] == str(seeded.planner_id)
        # Tick replay applied nothing twice.
        assert (await snapshot_configuration_state(s, seeded.goal_id))["agents"] == after_first_tick["agents"]
        proposal_cards = [row for row in await recovery_cards(s, seeded.goal_id)
                          if row.related_action_id == seeded.proposal_id]
        assert [(row.status, row.selected_option) for row in proposal_cards] == [("answered", "approve_fix")]


@pytest.mark.asyncio
async def test_approved_plan_author_fix_never_raises_a_second_fix_approval(
    sessions, tmp_path, monkeypatch, stub_decision, test_engine,  # noqa: F811
):
    seeded = await diagnosed_spr180(sessions, await started_spr180(sessions, tmp_path, monkeypatch, stub_decision))
    async with committed_client(test_engine, seeded.user_id) as client:
        response = await answer(client, seeded.project_id, seeded.goal_id, seeded.card_id,
                                selected_option="approve_fix")
        assert response.status_code == 200, response.text
    for _ in range(3):
        await tick(sessions, seeded.run_id)

    async with sessions() as s:
        proposal = await s.get(OrchestrationAction, seeded.proposal_id, populate_existing=True)
        assert proposal.status == "completed"
        assert len(fix_cards(await recovery_cards(s, seeded.goal_id))) == 1, "a second fix approval was raised"
        assert await active_warnings(s, seeded.goal_id, "recovery_blocked") == [], "resolved blocker still warns"
        assert [t.request["blocker_code"] for t in await actions(s, seeded.run_id, "recovery_trigger")] == [
            "plan_author_unavailable"]


@pytest.mark.asyncio
async def test_plan_author_fix_discloses_and_invalidates_a_collateral_goal_in_one_transaction(
    sessions, tmp_path, monkeypatch, stub_decision, test_engine,  # noqa: F811
):
    from huddleroom.services.orchestration_roster_mutation import INVALIDATED_PROCESS_STATUS

    seeded = await diagnosed_spr180(
        sessions, await started_spr180(sessions, tmp_path, monkeypatch, stub_decision, second_goal=True))
    async with sessions() as s:
        other_before = {kind: (await OrchestrationProcessService().get_current(s, seeded.other_goal_id, kind)).id
                        for kind in ("agent_definition_review", "team_hierarchy")}
    async with committed_client(test_engine, seeded.user_id) as client:
        card = await projected(client, seeded.project_id, seeded.goal_id, seeded.card_id)
        assert card["recovery"]["global_effect"] is True
        assert any("2 affected goals" in bullet for bullet in card["recovery"]["general_updates"]), (
            card["recovery"]["general_updates"])
        (call,) = card["recovery"]["exact_details"]["calls"]
        disclosed = set()
        for change in call["changes"]:
            if change["target_type"] == "configuration_fingerprint":
                disclosed.update(change["after"] if change["target_id"] == "aggregate" else [change["target_id"]])
        assert str(seeded.other_goal_id) in disclosed
        response = await answer(client, seeded.project_id, seeded.goal_id, seeded.card_id,
                                selected_option="approve_fix")
        assert response.status_code == 200, response.text

    arm_semantic_guard(monkeypatch)
    await tick(sessions, seeded.run_id)

    async with sessions() as s:
        assert await plan_author(s, seeded) == seeded.planner_id
        for kind in ("agent_definition_review", "team_hierarchy"):
            origin = await OrchestrationProcessService().get_current(s, seeded.goal_id, kind)
            assert origin.trigger_reason == "approved_recovery" and origin.status == "completed"
            other = await OrchestrationProcessService().get_current(s, seeded.other_goal_id, kind)
            assert other.id != other_before[kind] and other.status == INVALIDATED_PROCESS_STATUS
        assert not await OrchestrationService()._baseline_processes_ready_for_goal(
            s, seeded.other_goal_id, allow_heal=False)
        warnings = await active_warnings(s, seeded.other_goal_id)
        assert any(row.related_action_id == seeded.proposal_id for row in warnings)
        other_definition = await OrchestrationProcessService().get_current(s, seeded.other_goal_id,
                                                                           "agent_definition_review")
        assert str(seeded.planner_id) not in ((other_definition.outputs or {}).get("eligibility") or {})
        other_triggers = await actions(s, seeded.other_run_id, "recovery_trigger")
        assert len(other_triggers) == 1 and other_triggers[0].request["goal_id"] == str(seeded.other_goal_id)
        # The other goal never receives the origin's approval card.
        assert await recovery_cards(s, seeded.other_goal_id) == []


# --------------------------------------------------------------------------- SPR 184: eventless noop


@pytest.mark.asyncio
async def test_eventless_noop_enters_recovery_without_warning_loop(
    db_session, test_project, stub_decision, monkeypatch,
):
    from tests.test_orchestration_runtime_e2e import _authorized_run

    analyzer_calls = script_analyzer(monkeypatch, lambda: gap_assessment(uuid.uuid4()))
    stub_decision(lambda _ctx: {"action_type": "noop", "reason": "Waiting for the environment to recover.",
                                "wake_when": {"recheck_after_seconds": 60,
                                              "expected_result": "The environment recovers."}})
    service, goal, run = await _authorized_run(db_session, test_project)
    warnings_before = len(await active_warnings(db_session, goal.id))

    observed = []
    for _ in range(5):
        await service.tick(db_session, run.id)
        observed.append((len(await actions(db_session, run.id, "noop")),
                         len(await active_warnings(db_session, goal.id)),
                         len(await actions(db_session, run.id, "record_warning"))))

    triggers = await actions(db_session, run.id, "recovery_trigger")
    recovery_trigger_count = len(triggers)
    assert recovery_trigger_count == 1
    assert triggers[0].request["source_kind"] == "eventless_noop"
    assert triggers[0].request["retry_target"] == {"kind": "decision_loop"}
    noop_count = observed[-1][0]
    repeated_noop_count = max(noop_count - 1, 0)
    repeated_warning_count = observed[-1][1] - observed[1][1] + observed[-1][2]
    assert repeated_noop_count == repeated_warning_count == 0, observed
    assert len(analyzer_calls) == 1, "diagnosis ran once; the capability wait is not re-analysed every tick"
    assert await active_warnings(db_session, goal.id, "supervision_attention") == []
    assert len(await active_warnings(db_session, goal.id)) - warnings_before <= 1
    (gap_card,) = await recovery_cards(db_session, goal.id, status="pending")
    assert fix_cards([gap_card]) == [], "a capability gap is never an executable fix"


# --------------------------------------------------------------------------- capability-gap wait


@pytest.mark.asyncio
async def test_capability_gap_waits_for_the_named_human_input_then_rediagnoses_once(
    client, db_session, auth_headers, test_project, stub_decision, monkeypatch,
):
    from huddleroom.models.orchestration import OrchestrationDecision
    from huddleroom.services.orchestration_decision_dispatcher import OrchestrationDecisionDispatcher
    from tests.test_orchestration_runtime_e2e import _authorized_run

    stub_decision(lambda _ctx: {"action_type": "ask_human", "question": "Proceed?", "reason": "Unsure."})
    service, goal, run = await _authorized_run(db_session, test_project)
    analyzer_calls = script_analyzer(monkeypatch, gap_assessment(run.id))
    report = OrchestrationDecision(run_id=run.id, decision_type="coordination", input_snapshot={},
                                   validator_status="accepted",
                                   parsed_decision={"action_type": "recover_blocker",
                                                    "reason": "The release check cannot reach staging.",
                                                    "evidence_refs": []})
    db_session.add(report)
    await db_session.flush()
    await OrchestrationDecisionDispatcher(service).dispatch(db_session, run, report)

    for _ in range(3):
        await service.tick(db_session, run.id)

    assert len(analyzer_calls) == 1
    (card,) = await recovery_cards(db_session, goal.id, status="pending")
    item = await projected(client, test_project.id, goal.id, card.id, auth_headers)
    assert item["recovery"]["status"] == "unsupported"
    assert [option["key"] for option in item["options"]] == ["capability_provided"]
    assert "staging" in str(item["recovery"]).lower(), "the most specific evidenced missing capability is named"
    refused = await answer(client, test_project.id, goal.id, card.id, auth_headers, selected_option="approve_fix")
    assert refused.status_code in (400, 409)
    assert await actions(db_session, run.id, "recovery_apply") == []

    provided = await answer(client, test_project.id, goal.id, card.id, auth_headers,
                            selected_option="capability_provided")
    assert provided.status_code == 200, provided.text
    await service.tick(db_session, run.id)

    assert len(analyzer_calls) == 2, "the human input is new evidence: exactly one fresh diagnosis"
    await service.tick(db_session, run.id)
    assert len(analyzer_calls) == 2


# --------------------------------------------------------------------------- delegation


@pytest.mark.asyncio
async def test_delegated_remediation_is_admitted_awaited_verified_and_retried_once(
    client, db_session, auth_headers, test_project, monkeypatch,
):
    from huddleroom.services import orchestration_service as service_module
    from tests.test_orchestration_recovery_delegation import (
        HIERARCHY_RETRY, draft, executables, fail_process, seed, spy_build_command, trigger,
    )
    from tests.test_orchestration_recovery_execution import deliver, open_waits_for

    executables(monkeypatch)  # real CliAdapter command inspection; the executable lookup is faked
    built = spy_build_command(monkeypatch)
    no_session_dispatch(monkeypatch)
    rows = await seed(db_session, test_project)
    failed = await fail_process(db_session, rows, "team_hierarchy")
    assert await rows.service._baseline_readiness_reason(db_session, rows.goal.id) is not None
    script_analyzer(monkeypatch, draft(rows))
    svc = OrchestrationBlockerRecoveryService()
    trigger_action = await svc.create_trigger(db_session, trigger(rows, HIERARCHY_RETRY))

    proposal = await svc.diagnose(db_session, trigger_action.id)

    assert proposal.action_type == "recovery_proposal" and built, "qualification used the built CLI command"
    (call,) = proposal.request["calls"]
    assert call["tool"] == "orchestration.delegate_remediation"
    assert call["arguments"]["actual_execution_scope"]["unrestricted_tools"] is True
    (admission,) = [change["after"] for change in call["changes"] if change["target_type"] == "baseline_repair_admission"]
    assert admission["process_type"] == "team_hierarchy" and admission["gate_id"] == str(failed.id)
    (card,) = await recovery_cards(db_session, rows.goal.id, status="pending")
    item = await projected(client, test_project.id, rows.goal.id, card.id, auth_headers)
    assert any("unrestricted local tools" in bullet for bullet in item["recovery"]["general_updates"])

    approved = await answer(client, test_project.id, rows.goal.id, card.id, auth_headers, selected_option="approve_fix")
    assert approved.status_code == 200, approved.text
    assert await svc.resume(db_session, proposal.id) == "applying"

    (apply_wrapper,) = await actions(db_session, rows.run.id, "recovery_apply")
    receipt = apply_wrapper.dispatch_contract["tool_receipt_v1"]
    assert receipt["status"] == "waiting"
    (task_ref,) = [ref for ref in receipt["target_refs"] if ref["kind"] == "task"]
    task = await db_session.get(Task, uuid.UUID(task_ref["id"]))
    assert task.assigned_to == rows.agent.id
    assert await rows.service._baseline_readiness_reason(db_session, rows.goal.id) is not None, (
        "only the one named remediation was admitted; the baseline is still blocked")
    assert await open_waits_for(db_session, rows.run.id, task.id)
    goal_lock = service_module._SQLITE_GOAL_LOCKS.get(rows.goal.id)
    assert goal_lock is None or not goal_lock._lock.locked(), "no lock is held across the async wait"

    other = Task(project_id=test_project.id, title="unrelated", status="done")
    db_session.add(other)
    await db_session.flush()
    await deliver(db_session, rows.run, "task.status_changed", other.id)
    assert await svc.resume(db_session, proposal.id) == "applying", "an unrelated event does not resume it"
    assert await actions(db_session, rows.run.id, "recovery_verify") == []

    task.status = "done"
    await db_session.flush()
    await deliver(db_session, rows.run, "task.status_changed", task.id)  # synthetic completion event
    assert await svc.resume(db_session, proposal.id) == "retrying", "verified; the retry waits for the process"
    (verify_wrapper,) = await actions(db_session, rows.run.id, "recovery_verify")
    assert verify_wrapper.status == "completed"
    assert (await projected(client, test_project.id, rows.goal.id, card.id, auth_headers))["recovery"][
        "status"] == "retrying"

    # The repaired hierarchy completes (normal process path); the one sealed retry observes it.
    failed.status, failed.outputs = "completed", {"gates": {"team_structure_reviewed": True}}
    await db_session.flush()
    assert await svc.resume(db_session, proposal.id) == "resolved"
    assert await svc.resume(db_session, proposal.id) == "resolved"
    assert len(await actions(db_session, rows.run.id, "recovery_retry")) == 1
    assert len([row for row in await actions(db_session, rows.run.id, "create_delegation_task")
                if row.idempotency_key.startswith("recovery:")]) == 1
    assert await open_waits_for(db_session, rows.run.id, task.id) == []
    assert len(fix_cards(await recovery_cards(db_session, rows.goal.id))) == 1, "no second approval"


# --------------------------------------------------------------------------- materially different lineage


@pytest.mark.asyncio
async def test_failed_fix_permits_only_a_materially_different_retry_with_distinct_identities(
    client, db_session, auth_headers, test_project, monkeypatch,
):
    from huddleroom.services.orchestration_blocker_recovery import RecoveryDiagnosisFailed
    from tests.test_orchestration_recovery_execution import Executors, draft, retry_task_call, seed, trigger, verify_call

    rows = await seed(db_session, test_project)
    first_draft = draft(rows, verify_call(rows))
    script_analyzer(monkeypatch, first_draft, first_draft, draft(rows, retry_task_call(rows), verify_call(rows)))
    executors = Executors(monkeypatch, rows, plan={"request_verification": ["reject"]})
    svc = OrchestrationBlockerRecoveryService()
    first = await svc.diagnose(db_session, (await svc.create_trigger(db_session, trigger(rows))).id)
    (first_card,) = await recovery_cards(db_session, rows.goal.id, status="pending")
    assert (await answer(client, test_project.id, rows.goal.id, first_card.id, auth_headers,
                         selected_option="approve_fix")).status_code == 200

    assert await svc.resume(db_session, first.id) == "failed"
    follow_up = [row for row in await actions(db_session, rows.run.id, "recovery_trigger") if row.status == "reserved"]
    assert len(follow_up) == 1 and follow_up[0].dispatch_contract["supersedes_proposal_action_id"] == str(first.id)

    with pytest.raises(RecoveryDiagnosisFailed) as repeated:  # the same proposal against the same state
        await svc.diagnose(db_session, follow_up[0].id)
    assert "duplicate" in repeated.value.code
    assert fix_cards(await recovery_cards(db_session, rows.goal.id, status="pending")) == []

    second = await svc.diagnose(db_session, follow_up[0].id)  # materially different: retry the task first
    assert second.id != first.id
    assert second.dispatch_contract["lineage_id"] == first.dispatch_contract["lineage_id"]
    assert second.dispatch_contract["attempt"] == 2
    (second_card,) = await recovery_cards(db_session, rows.goal.id, status="pending")
    assert (await answer(client, test_project.id, rows.goal.id, second_card.id, auth_headers,
                         selected_option="approve_fix")).status_code == 200
    final = await svc.resume(db_session, second.id)

    assert final in {"retrying", "resolved"}, final
    first_pid, second_pid = first.request["proposal_id"], second.request["proposal_id"]
    verification_keys = executors.keys("request_verification")
    assert any(key.startswith(f"recovery:{first_pid}:apply:") for key in verification_keys)
    assert any(key.startswith(f"recovery:{second_pid}:apply:") and key.endswith(":effect")
               for key in verification_keys), "the same domain request runs again under the new proposal identity"
    wrappers = [row.idempotency_key for row in await actions(db_session, rows.run.id, "recovery_apply")]
    assert len(wrappers) == len(set(wrappers)) == 3


# --------------------------------------------------------------------------- reset at each phase


async def reset_goal(sessions, seeded):  # noqa: F811
    async with sessions() as s:
        _goal, run = await OrchestrationService().reset_goal(s, seeded.project_id, seeded.goal_id)
        await s.commit()
        return run.id


async def assert_fresh_after_reset(sessions, seeded, new_run_id):  # noqa: F811
    async with sessions() as s:
        assert await s.get(OrchestrationRun, seeded.run_id) is None
        for kind in ("recovery_trigger", "recovery_proposal", "recovery_apply", "recovery_verify", "recovery_retry"):
            assert await actions(s, new_run_id, kind) == [], kind
        assert await recovery_cards(s, seeded.goal_id) == []
        assert await active_warnings(s, seeded.goal_id, "recovery_blocked") == []
    await tick(sessions, new_run_id)
    async with sessions() as s:
        for kind in ("recovery_proposal", "recovery_apply", "recovery_retry"):
            assert await actions(s, new_run_id, kind) == [], kind
        assert await recovery_cards(s, seeded.goal_id) == []


class Reservation:
    """A reset that has signalled (SPR 175/176 reservation) but has not yet removed the run."""

    def __init__(self, goal_id):
        self.goal_id = goal_id

    def __enter__(self):
        runtime._resetting[self.goal_id] = runtime._resetting.get(self.goal_id, 0) + 1
        return self

    def __exit__(self, *exc):
        runtime._resetting[self.goal_id] -= 1
        if not runtime._resetting[self.goal_id]:
            runtime._resetting.pop(self.goal_id)


async def attempt(coro):
    try:
        return await coro
    except HTTPException as exc:
        return exc


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["diagnosis", "approval", "apply", "verification", "retry"])
async def test_reset_during_each_recovery_phase_never_writes_into_the_fresh_run(
    sessions, tmp_path, monkeypatch, stub_decision, test_engine, phase,  # noqa: F811
):
    seeded = await started_spr180(sessions, tmp_path, monkeypatch, stub_decision)
    if phase == "diagnosis":
        draft = proposal_draft(seeded, configuration_arguments(seeded))

        def reset_signalled_while_analysing():
            runtime._resetting[seeded.goal_id] = runtime._resetting.get(seeded.goal_id, 0) + 1
            return draft

        calls = script_analyzer(monkeypatch, reset_signalled_while_analysing)
        try:
            await tick(sessions, seeded.run_id)
            assert len(calls) == 1, "the reset was signalled while the analyzer was running"
            async with sessions() as s:
                assert await actions(s, seeded.run_id, "recovery_proposal") == []
                assert await recovery_cards(s, seeded.goal_id) == []
        finally:
            runtime._resetting.pop(seeded.goal_id, None)
    else:
        await diagnosed_spr180(sessions, seeded)
        svc = OrchestrationBlockerRecoveryService()
        if phase == "approval":
            with Reservation(seeded.goal_id):
                async with committed_client(test_engine, seeded.user_id) as client:
                    response = await answer(client, seeded.project_id, seeded.goal_id, seeded.card_id,
                                            selected_option="approve_fix")
            assert response.status_code == 409, response.text
            async with sessions() as s:
                (card,) = await recovery_cards(s, seeded.goal_id)
                assert card.status == "pending"
                assert await actions(s, seeded.run_id, "recovery_apply") == []
        else:
            async with committed_client(test_engine, seeded.user_id) as client:
                response = await answer(client, seeded.project_id, seeded.goal_id, seeded.card_id,
                                        selected_option="approve_fix")
                assert response.status_code == 200, response.text
            arm_semantic_guard(monkeypatch)
            async with sessions() as s:
                proposal = await s.get(OrchestrationAction, seeded.proposal_id)
                if phase in {"verification", "retry"}:
                    assert await svc.execute_next_call(s, proposal) is not None
                    await s.commit()
                if phase == "retry":
                    results = await svc.verify_proposal(s, proposal)
                    assert [item.status for item in results] == ["passed"] * len(results)
                    await s.commit()
            async with sessions() as s:
                proposal = await s.get(OrchestrationAction, seeded.proposal_id)
                step = {"apply": svc.execute_next_call, "verification": svc.verify_proposal,
                        "retry": svc.retry_blocked_work}[phase]
                with Reservation(seeded.goal_id):
                    outcome = await attempt(step(s, proposal))
                    await s.rollback()
                    tick_result = await tick(sessions, seeded.run_id)
            assert isinstance(outcome, HTTPException) and outcome.status_code == 409, outcome
            assert tick_result.get("interrupted") is True, tick_result
            async with sessions() as s:
                pid = (await s.get(OrchestrationAction, seeded.proposal_id)).request["proposal_id"]
                marker = {"apply": "apply", "verification": "verify", "retry": "retry"}[phase]
                fenced = [row for kind in ("recovery_apply", "recovery_verify", "recovery_retry")
                          for row in await actions(s, seeded.run_id, kind)
                          if row.idempotency_key.startswith(f"recovery:{pid}:{marker}") and row.status == "completed"]
                assert fenced == [], f"{phase} wrote while the reset was signalled"
                if phase == "apply":
                    assert (await s.get(Agent, seeded.planner_id, populate_existing=True)).description == ""

    new_run_id = await reset_goal(sessions, seeded)
    await assert_fresh_after_reset(sessions, seeded, new_run_id)
    if phase != "diagnosis":  # the old approval can never authorize work in the fresh run
        async with committed_client(test_engine, seeded.user_id) as client:
            stale = await answer(client, seeded.project_id, seeded.goal_id, seeded.card_id,
                                 selected_option="approve_fix")
        assert stale.status_code in (404, 409), stale.text


@pytest.mark.asyncio
async def test_reset_during_an_async_delegation_wait_discards_the_remediation(
    db_session, test_project, test_user, monkeypatch,
):
    from tests.test_orchestration_recovery_execution import deliver, delegation_flow

    no_session_dispatch(monkeypatch)
    flow = await delegation_flow(db_session, test_project, test_user, monkeypatch)
    receipt = await flow.svc.execute_next_call(db_session, flow.proposal)
    assert receipt.status == "waiting"
    task = await db_session.get(Task, uuid.UUID(next(ref["id"] for ref in receipt.target_refs if ref["kind"] == "task")))
    old_run_id = flow.rows.run.id

    with Reservation(flow.rows.goal.id):  # the delegated task completes while the reset is signalled
        task.status = "done"
        await db_session.flush()
        await deliver(db_session, flow.rows.run, "task.status_changed", task.id)
        outcome = await attempt(flow.svc.resume(db_session, flow.proposal.id))
    assert isinstance(outcome, HTTPException) and outcome.status_code == 409, outcome
    assert await actions(db_session, old_run_id, "recovery_verify") == []
    assert await actions(db_session, old_run_id, "recovery_retry") == []

    _goal, new_run = await OrchestrationService().reset_goal(db_session, test_project.id, flow.rows.goal.id)

    for kind in ("recovery_trigger", "recovery_proposal", "recovery_apply", "recovery_verify", "recovery_retry"):
        assert await actions(db_session, new_run.id, kind) == [], kind
    assert await recovery_cards(db_session, flow.rows.goal.id) == []
    waits = (await db_session.scalars(select(OrchestrationWait).where(OrchestrationWait.run_id == new_run.id))).all()
    assert waits == []
    assert await db_session.scalar(select(func.count(OrchestrationAction.id)).where(
        OrchestrationAction.run_id == old_run_id)) == 0


@pytest.mark.asyncio
async def test_team_configuration_apply_under_a_goal_lock_is_left_for_the_next_pass(
    sessions, tmp_path, monkeypatch, stub_decision, test_engine,  # noqa: F811
):
    """F5: resuming a roster-first apply while a caller (e.g. a tick) holds the goal lock is a no-op, not a
    RuntimeError; the next lock-free pass applies it."""
    seeded = await diagnosed_spr180(sessions, await started_spr180(sessions, tmp_path, monkeypatch, stub_decision))
    async with committed_client(test_engine, seeded.user_id) as client:
        response = await answer(client, seeded.project_id, seeded.goal_id, seeded.card_id,
                                selected_option="approve_fix")
        assert response.status_code == 200, response.text
    arm_semantic_guard(monkeypatch)
    svc = OrchestrationBlockerRecoveryService()
    async with sessions() as s:
        proposal = await s.get(OrchestrationAction, seeded.proposal_id)
        async with OrchestrationService()._lock_goal_for_baseline_transition(s, seeded.goal_id):
            assert await svc.execute_next_call(s, proposal) is None
        assert [row for row in await actions(s, seeded.run_id, "recovery_apply") if row.status == "completed"] == []
        await s.rollback()
    async with sessions() as s:
        proposal = await s.get(OrchestrationAction, seeded.proposal_id)
        receipt = await svc.execute_next_call(s, proposal)
        assert receipt is not None and receipt.status == "applied", receipt
