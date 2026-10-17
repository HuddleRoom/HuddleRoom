import uuid

import pytest

from huddleroom.services.orchestration_condition import (
    CONDITIONS,
    ConditionError,
    is_reconcilable,
    validate_condition,
)

T1 = "6f1c1f0e-1d2b-4c3a-9e8f-0a1b2c3d4e5f"


@pytest.mark.parametrize("condition", sorted(CONDITIONS))
def test_valid_condition_is_normalized(condition):
    assert validate_condition({"condition": condition, "subject": f" {T1.upper()} "}) == {
        "condition": condition,
        "subject": T1,
    }


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "session_running",
        {"condition": "whenever", "subject": T1},
        {"condition": "session_running"},
        {"condition": "plan_missing", "subject": "  "},
        {"condition": "gate_unverified", "subject": 3},
    ],
)
def test_invalid_condition_rejected(raw):
    with pytest.raises(ConditionError):
        validate_condition(raw)
    assert not is_reconcilable(raw)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"condition": "session_running", "subject": "t1"}, "session_running subject must be a UUID"),
        ({"condition": "valid_result_missing", "subject": "task 7"}, "valid_result_missing subject must be a UUID"),
        ({"condition": "gate_unverified", "subject": "gate:x"}, "gate_unverified subject must be a UUID"),
        ({"condition": "plan_missing", "subject": "run-1"}, "plan_missing subject run must be a UUID"),
        ({"condition": "plan_missing", "subject": f"{T1}:v2"}, "version must be digits"),
        ({"condition": "plan_missing", "subject": f"{T1}:"}, "version must be digits"),
        ({"condition": "plan_missing", "subject": f"{T1}:\u00b2"}, "version must be digits"),
    ],
)
def test_never_holding_subject_is_rejected_with_its_reason(raw, message):
    with pytest.raises(ConditionError, match=message):
        validate_condition(raw)


def test_plan_subject_accepts_a_run_with_an_optional_version():
    assert validate_condition({"condition": "plan_missing", "subject": T1})["subject"] == T1
    assert validate_condition({"condition": "plan_missing", "subject": f"{T1.upper()}:03"})["subject"] == f"{T1}:3"


@pytest.mark.parametrize("kind", ["budget", "permission", "provider", "Owner_Decision"])
def test_owner_decisions_never_reconcilable(kind):
    raw = {"condition": "session_running", "subject": T1, "kind": kind}
    assert not is_reconcilable(raw)


@pytest.mark.parametrize("subject", ["budget", " Permission ", "provider"])
def test_owner_decision_subject_rejected(subject):
    with pytest.raises(ConditionError):
        validate_condition({"condition": "session_running", "subject": subject})


@pytest.mark.parametrize("kw", [{"action_kind": "budget"}, {"category": "permission_required"}])
def test_reconcile_time_action_context_blocks_owner_decisions(kw):
    stored = validate_condition({"condition": "valid_result_missing", "subject": T1})
    assert is_reconcilable(stored)
    assert not is_reconcilable(stored, **kw)
    assert is_reconcilable(stored, action_kind="record_warning", category="other")


# --- Reconciliation through the real record/ask actions, ingestion and ticks ---

import json  # noqa: E402
from datetime import timedelta  # noqa: E402

from sqlalchemy import select  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402

from huddleroom.models.base import _utcnow  # noqa: E402
from huddleroom.models.event_log import EventLog  # noqa: E402
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGate, OrchestrationWait  # noqa: E402
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision, OrchestrationWarning  # noqa: E402
from huddleroom.models.session import Session  # noqa: E402
from huddleroom.services.event_bus import emit_event_once  # noqa: E402
from huddleroom.services.orchestration_decision_validator import validate_orchestration_decision  # noqa: E402
from huddleroom.services.session_sync import sync_task_from_session  # noqa: E402
from tests.test_orchestration_final_summary_request import runnable_workspace  # noqa: E402,F401  (autouse)
from tests.test_orchestration_plan_ingestion import _db_output, _planning_run  # noqa: E402
from tests.test_orchestration_result_correction import (  # noqa: E402
    MISSING_FIELDS, VALID, _complete, _corrections, _setup as _result_setup,
)


@pytest.fixture
async def db_session(test_engine):
    async with AsyncSession(test_engine, expire_on_commit=False) as db:
        yield db
        await db.rollback()


async def _owned_setup_call(db, setup, *args, **kwargs):
    from huddleroom.services.orchestration_service import OrchestrationService

    original = OrchestrationService.tick
    async def owned_setup_tick(self, current_db, *args, **tick_kwargs):
        root = current_db.sync_session.get_transaction()
        if (current_db is db and not current_db.in_nested_transaction()
                and (root is None or root.origin.name == "AUTOBEGIN")):
            await current_db.flush()
            await current_db.commit()
        return await original(self, current_db, *args, **tick_kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(OrchestrationService, "tick", owned_setup_tick)
        return await setup(db, *args, **kwargs)


async def _setup(db, project, **kwargs):
    return await _owned_setup_call(db, _result_setup, project, **kwargs)


@pytest.fixture(autouse=True)
def no_live_decisions(stub_decision):
    stub_decision(lambda _context: {"action_type": "noop"})


async def _warn(service, db, run, message, condition=None, warning_type="model_attention"):
    request = {"action_type": "record_warning", "warning_type": warning_type, "severity": "warning", "message": message}
    if condition is not None:
        request["condition"] = condition
    action = await service.execute_record_warning_action(db, run.id, request, f"run:{run.id}:test-warning:{uuid.uuid4()}")
    return await db.get(OrchestrationWarning, action.target_id)


async def _ask(service, db, run, question, condition=None, reason=None):
    request = {"action_type": "ask_human", "question": question, "subject": question}
    if condition is not None:
        request["condition"] = condition
    if reason is not None:
        request["reason"] = reason
    action = await service.execute_ask_human_action(db, run.id, request, f"run:{run.id}:test-ask:{uuid.uuid4()}")
    return await db.get(OrchestrationAuthorityDecision, action.target_id)


def test_decision_validator_accepts_only_declared_conditions():
    base = {"action_type": "record_warning", "warning_type": "x", "severity": "warning", "message": "m"}
    assert validate_orchestration_decision({**base, "condition": {"condition": "plan_missing", "subject": T1}}).accepted
    assert not validate_orchestration_decision({**base, "condition": {"condition": "soon", "subject": T1}}).accepted
    rejected = validate_orchestration_decision({**base, "condition": {"condition": "plan_missing", "subject": "r"}})
    assert not rejected.accepted and "must be a UUID" in rejected.rejection_reason
    ask = {"action_type": "ask_human", "question": "q"}
    assert validate_orchestration_decision({**ask, "condition": {"condition": "gate_unverified", "subject": T1}}).accepted
    assert not validate_orchestration_decision({**ask, "condition": {"condition": "gate_unverified", "subject": "budget"}}).accepted


async def test_late_plan_clears_its_warning_and_question_but_never_owner_or_undeclared_items(db_session, test_project):
    from huddleroom.services.orchestration_plan_ingestion import ingest

    service, goal, run, task, session = await _planning_run(db_session, test_project.id)
    version = run.plan_state["planning_version"]
    plan_missing = {"condition": "plan_missing", "subject": f"{run.id}:{version}"}
    warning = await _warn(service, db_session, run, "No plan yet.", plan_missing, "plan_missing")
    question = await _ask(service, db_session, run, "Should planning continue?", plan_missing)
    owner_budget = await _ask(service, db_session, run, "Raise the budget?", plan_missing, reason="budget_exhausted")
    escalation = await service._escalate_owner(
        db_session, goal, run, "budget_exhausted", "plan", ["spent"], "Raise the budget for planning?",
    )
    escalated = await db_session.get(OrchestrationAuthorityDecision, escalation.target_id)
    undeclared = await _warn(service, db_session, run, "The plan looks slow.")

    await service._reconcile_conditions(db_session, run)
    assert warning.active and question.status == "pending"

    artifact = await ingest(db_session, goal, run, task, session.id, version, _db_output())
    await service.execute_accept_plan_action(
        db_session, run_id=run.id, request={"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
        idempotency_key=f"run:{run.id}:kind:accept_plan",
    )
    await service._reconcile_conditions(db_session, run)
    await service._reconcile_conditions(db_session, run)  # idempotent

    for row in (warning, question, owner_budget, escalated, undeclared):
        await db_session.refresh(row)
    assert not warning.active and warning.resolved_by == "system"
    assert question.status == "cancelled"
    assert owner_budget.status == "pending" and escalated.status == "pending"
    assert undeclared.active


async def test_rejected_output_keeps_result_missing_until_the_corrected_result_is_accepted(db_session, test_project):
    service, goal, run, agent, task = await _setup(db_session, test_project)
    task.status = "in_progress"
    running = Session(
        project_id=task.project_id, task_id=task.id, agent_id=agent.id, adapter_type="api", status="running",
        input_context={}, origin="orchestrator",
        metadata_={"orchestration": {"action_id": task.metadata_["orchestration"]["action_id"]}},
    )
    db_session.add(running)
    await db_session.flush()
    supervision = service.supervision
    await supervision.reconcile_local(db_session, goal, run)
    wait = await db_session.scalar(select(OrchestrationWait).where(
        OrchestrationWait.run_id == run.id, OrchestrationWait.wait_key.like(f"%session:{running.id}%"),
    ))
    wait.due_recheck_at = _utcnow() - timedelta(seconds=1)
    fired = await supervision.reconcile_local(db_session, goal, run)
    assert fired["outcome"] == "due_fallback"
    fallback = await db_session.get(OrchestrationAction, uuid.UUID(fired["action_id"]))
    assert fallback.request["condition"] == {"condition": "session_running", "subject": str(task.id)}
    session_warning = await db_session.get(OrchestrationWarning, fallback.target_id)
    result_warning = await _warn(
        service, db_session, run, "No valid result yet.", {"condition": "valid_result_missing", "subject": str(task.id)},
    )

    # The session ends with output that fails validation: the session condition holds, the result one does not.
    running.status, running.output, running.ended_at = "completed", MISSING_FIELDS, _utcnow()
    await db_session.flush()
    await sync_task_from_session(db_session, running)
    key = f"condition-test:{running.id}"
    await emit_event_once(db_session, task.project_id, "session.completed", {
        "session_id": str(running.id), "task_id": str(task.id),
    }, dedup_key=key)
    await service._ingest_evidence_from_event(
        db_session, run, await db_session.scalar(select(EventLog).where(EventLog.dedup_key == key)),
    )
    await db_session.commit()  # Publish this test's completed-session/event setup before the owned tick.
    await service.tick(db_session, run.id)
    await db_session.refresh(session_warning)
    await db_session.refresh(result_warning)
    assert not session_warning.active and session_warning.resolved_by == "system"
    assert result_warning.active

    [correction] = await _corrections(db_session, task)
    await _complete(db_session, service, run, correction, json.dumps(VALID))
    await db_session.commit()
    await service.tick(db_session, run.id)
    await db_session.refresh(result_warning)
    assert not result_warning.active and result_warning.resolved_by == "system"


async def test_gate_condition_clears_only_on_acceptance_and_shared_undeclared_item_stays(db_session, test_project):
    service, goal, run, _agent, _task = await _setup(db_session, test_project)
    gate = OrchestrationGate(run_id=run.id, success_criterion_key="done", gate_type="work_completed")
    db_session.add(gate)
    await db_session.flush()
    unverified = {"condition": "gate_unverified", "subject": str(gate.id)}
    cleared = await _warn(service, db_session, run, "Gate not verified.", unverified, "gate_unverified")
    shared = await _warn(service, db_session, run, "Gate still open.", unverified)
    # Warnings of one type dedupe into one row: an undeclared origin keeps the shared row open.
    assert (await _warn(service, db_session, run, "Gate still open, says the model.")).id == shared.id

    gate.status = "failed"
    await service._reconcile_conditions(db_session, run)
    assert cleared.active

    gate.status = "accepted"
    await service._reconcile_conditions(db_session, run)
    await db_session.refresh(cleared)
    await db_session.refresh(shared)
    assert not cleared.active and shared.active


async def test_one_malformed_item_is_skipped_and_the_rest_still_reconcile(db_session, test_project):
    service, goal, run, _agent, task = await _setup(db_session, test_project)
    gate = OrchestrationGate(run_id=run.id, success_criterion_key="done", gate_type="work_completed", status="accepted")
    db_session.add(gate)
    await db_session.flush()
    broken = await _warn(
        service, db_session, run, "No result yet.", {"condition": "valid_result_missing", "subject": str(task.id)},
        "result_missing",
    )
    cleared = await _warn(
        service, db_session, run, "Gate not verified.", {"condition": "gate_unverified", "subject": str(gate.id)},
        "gate_unverified",
    )
    # A corrupt correction ledger on the task's consumed marker makes its result check raise.
    marker = OrchestrationAction(
        run_id=run.id, idempotency_key=f"run:{run.id}:kind:report_consumed:task:{task.id}", action_type="report_consumed",
        status="completed", request={"result_correction": {"status": "corrected", "correction_task_ids": ["not-a-uuid"]}},
    )
    db_session.add(marker)
    await db_session.flush()

    await db_session.commit()  # Publish the intentionally corrupt ledger and warning fixtures.
    await service.tick(db_session, run.id)

    await db_session.refresh(broken)
    await db_session.refresh(cleared)
    assert broken.active and not cleared.active


async def _accepted_plan(db, project_id, *, version=1):
    from huddleroom.services.orchestration_plan_ingestion import ingest

    service, goal, run, task, session = await _planning_run(db, project_id)
    run.plan_state = {**run.plan_state, "planning_version": version}
    task.metadata_ = {**task.metadata_, "orchestration_plan": {
        **task.metadata_["orchestration_plan"], "planning_version": version,
    }}
    artifact = await ingest(db, goal, run, task, session.id, version, _db_output())
    action = await service.execute_accept_plan_action(
        db, run.id, {"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
        f"run:{run.id}:kind:accept_plan",
    )
    return service, goal, run, task, artifact, action


@pytest.mark.parametrize("damage", [
    "earlier_generation", "later_generation", "other_run_subject", "missing_artifact_id",
    "nonexistent_artifact", "wrong_artifact_task", "wrong_artifact_project", "deleted_artifact",
    "wrong_task_run", "wrong_task_goal", "wrong_artifact_version", "wrong_output_hash",
    "missing_snapshot", "wrong_snapshot_artifact", "wrong_snapshot_fingerprint", "incomplete_acceptance",
])
async def test_spr208_invalid_persisted_plan_retains_its_warning_and_question(db_session, test_project, damage):
    service, goal, run, task, artifact, acceptance = await _accepted_plan(db_session, test_project.id, version=2)
    subject = f"{run.id}:2"
    if damage == "earlier_generation":
        subject = f"{run.id}:1"
    elif damage == "later_generation":
        subject = f"{run.id}:3"
    elif damage == "other_run_subject":
        subject = f"{uuid.uuid4()}:2"
    elif damage == "missing_artifact_id":
        run.plan_state = {k: v for k, v in run.plan_state.items() if k != "accepted_artifact_id"}
    elif damage == "nonexistent_artifact":
        run.plan_state = {**run.plan_state, "accepted_artifact_id": str(uuid.uuid4())}
    elif damage == "wrong_artifact_task":
        artifact.linked_task_id = uuid.uuid4()
    elif damage == "wrong_artifact_project":
        artifact.project_id = uuid.uuid4()
    elif damage == "deleted_artifact":
        artifact.status = "deleted"
    elif damage in {"wrong_task_run", "wrong_task_goal"}:
        key = "run_id" if damage == "wrong_task_run" else "goal_id"
        task.metadata_ = {**task.metadata_, "orchestration": {**task.metadata_["orchestration"], key: str(uuid.uuid4())}}
    elif damage == "wrong_artifact_version":
        artifact.metadata_ = {**artifact.metadata_, "planning_version": 1}
    elif damage == "wrong_output_hash":
        run.plan_state = {**run.plan_state, "plan_output_hash": "unpersisted-output"}
    elif damage == "missing_snapshot":
        run.plan_state = {k: v for k, v in run.plan_state.items() if k != "accepted_plan_snapshot"}
    elif damage in {"wrong_snapshot_artifact", "wrong_snapshot_fingerprint"}:
        key = "artifact_id" if damage == "wrong_snapshot_artifact" else "fingerprint"
        run.plan_state = {**run.plan_state, "accepted_plan_snapshot": {
            **run.plan_state["accepted_plan_snapshot"], key: str(uuid.uuid4()),
        }}
    elif damage == "incomplete_acceptance":
        acceptance.status = "reserved"
    condition = {"condition": "plan_missing", "subject": subject}
    warning = await _warn(service, db_session, run, "Matching plan is still missing.", condition, "plan_missing")
    question = await _ask(service, db_session, run, "Supply the matching plan?", condition)
    await db_session.flush()

    await service._reconcile_conditions(db_session, run)

    await db_session.refresh(warning)
    await db_session.refresh(question)
    assert warning.active, damage
    assert warning.resolved_at is None and warning.resolved_by is None
    assert question.status == "pending", damage


@pytest.mark.parametrize("unversioned", [False, True])
async def test_spr208_resolution_history_replay_blockers_and_api_projection(
    db_session, test_project, client, unversioned,
):
    from sqlalchemy import func

    service, goal, run, _task, _artifact, _acceptance = await _accepted_plan(db_session, test_project.id)
    condition = {"condition": "plan_missing", "subject": str(run.id) if unversioned else f"{run.id}:1"}
    action = await service.execute_record_warning_action(db_session, run.id, {
        "action_type": "record_warning", "warning_type": "plan_missing", "severity": "blocker",
        "message": "Matching plan missing.", "condition": condition,
    }, f"spr208:blocker:{run.id}")
    warning = await db_session.get(OrchestrationWarning, action.target_id)
    owner = await _ask(service, db_session, run, "Approve the budget?", condition, reason="budget_exhausted")
    other = await _warn(service, db_session, run, "Independent warning.")
    unrelated_blocker = {"kind": "independent-review", "reason": "Owner approval pending"}
    run.active_blockers = [*run.active_blockers, unrelated_blocker]
    assert any(b["kind"] == f"warning:{warning.id}" for b in run.active_blockers)
    url = f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}"
    goals_url = f"/api/v1/projects/{test_project.id}/orchestration/goals"
    before = await client.get(goals_url)
    assert before.status_code == 200, before.text
    before_count = next(g["needs_you_count"] for g in before.json()["items"] if g["id"] == str(goal.id))

    await service._reconcile_conditions(db_session, run)
    await db_session.refresh(warning)
    assert not warning.active and warning.resolved_by == "system"
    assert warning.resolved_reason == "Condition satisfied: plan_missing" and warning.resolved_at is not None
    resolved_at = warning.resolved_at
    events = await db_session.scalar(select(func.count()).select_from(EventLog))
    assert await service._reconcile_conditions(db_session, run) == 0
    await db_session.refresh(warning)
    assert warning.resolved_at == resolved_at
    assert await db_session.scalar(select(func.count()).select_from(EventLog)) == events
    assert await db_session.get(OrchestrationWarning, warning.id) is warning
    assert run.active_blockers == [unrelated_blocker]
    await db_session.refresh(owner)
    await db_session.refresh(other)
    assert owner.status == "pending" and other.active

    active = await client.get(f"{url}/warnings?active_only=true")
    history = await client.get(f"{url}/warnings")
    pending = await client.get(f"{url}/decisions?status=pending")
    checkpoint = await client.get(f"{url}/decisions/checkpoint")
    detail = await client.get(url)
    after = await client.get(goals_url)
    assert active.status_code == history.status_code == pending.status_code == 200
    assert str(warning.id) not in {w["id"] for w in active.json()}
    assert str(other.id) in {w["id"] for w in active.json()}
    historical = next(w for w in history.json() if w["id"] == str(warning.id))
    assert historical["resolved_by"] == "system" and historical["resolved_reason"] == warning.resolved_reason
    assert str(owner.id) in {q["id"] for q in pending.json()}
    assert checkpoint.status_code == detail.status_code == 200
    assert str(owner.id) in {q["id"] for q in checkpoint.json()["items"]}
    # This endpoint counts both active warnings and their active blocker mirrors.
    assert after.status_code == 200
    assert next(g["needs_you_count"] for g in after.json()["items"] if g["id"] == str(goal.id)) == before_count - 2
    assert all(b["kind"] != f"warning:{warning.id}" for b in detail.json()["run"]["active_blockers"])


@pytest.mark.parametrize("retained_kind", ["undeclared_owner", "typed_unsatisfied"])
async def test_spr208_retained_first_batch_does_not_starve_later_warnings_or_questions(
    db_session, test_project, retained_kind,
):
    service, _goal, run, _task, _artifact, _acceptance = await _accepted_plan(db_session, test_project.id)
    satisfied = {"condition": "plan_missing", "subject": f"{run.id}:1"}
    unsatisfied = {"condition": "plan_missing", "subject": f"{run.id}:2"}
    retained = []
    old = _utcnow() - timedelta(days=1)
    for index in range(51):
        condition = unsatisfied if retained_kind == "typed_unsatisfied" else None
        warning = await _warn(service, db_session, run, f"Retained {index}", condition, f"retained_{index}")
        question = await _ask(service, db_session, run, f"Retained question {index}", condition)
        warning.created_at = old
        question.asked_at = old
        retained.extend((warning, question))
    warning = await _warn(service, db_session, run, "Later matching plan warning", satisfied, "plan_missing")
    question = await _ask(service, db_session, run, "Later matching question", satisfied)
    run.supervision_state = {**(run.supervision_state or {}), "spr208_unrelated": {"preserve": True}}
    await db_session.flush()

    # Re-loading the run between bounded passes forbids process-local-only progress.
    for _ in range(3):
        assert await service._reconcile_conditions(db_session, run) <= 100
        await db_session.flush()
        await db_session.refresh(run)

    await db_session.refresh(warning)
    await db_session.refresh(question)
    assert not warning.active and question.status == "cancelled"
    assert run.supervision_state["spr208_unrelated"] == {"preserve": True}
    for row in retained:
        await db_session.refresh(row)
        assert row.active if isinstance(row, OrchestrationWarning) else row.status == "pending"


@pytest.mark.parametrize("stage", ["original", "pending_replan", "accepted_replan"])
async def test_spr208_roadmap_accepted_generation_survives_replanning(
    db_session, test_project, test_user, stage,
):
    from huddleroom.models.task import Task
    from huddleroom.services.orchestration_plan_ingestion import ingest
    from tests.test_orchestration_roadmap_plans import (
        _approve, _roadmap_acceptance, _request_replan, _roadmap_task, _roadmap_fingerprint,
        _accept_replan_after_selected_authority,
    )

    service, goal, run, _manual, approval, _fingerprint = await _owned_setup_call(
        db_session, _roadmap_acceptance, test_project, test_user,
    )
    task = await db_session.get(Task, uuid.UUID(run.plan_state["planning_task_id"]))
    # This helper requests the task before switching its goal to roadmap.
    contract = task.metadata_["orchestration_contract"]
    task.metadata_ = {**task.metadata_, "orchestration_contract": {
        **contract, "report_schema": {**contract["report_schema"], "plan_kind": "roadmap"},
    }}
    output = json.dumps({"schema_version": 1, "plan_kind": "roadmap", "summary": "Roadmap",
                         "plan_items": [{**_roadmap_task(), "why_runnable": "Unstarted work."}]})
    artifact = await ingest(db_session, goal, run, task, None, run.plan_state["planning_version"], output)
    approval.decision_key = f"roadmap_plan:{_roadmap_fingerprint(service, goal, artifact.metadata_['plan_items'])}"
    await _approve(db_session, goal, approval)
    await service.execute_accept_plan_action(
        db_session, run.id, {"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
        f"spr208:roadmap:{run.id}:first",
    )
    accepted_version = run.plan_state["planning_version"]
    assert "accepted_plan_snapshot" not in run.plan_state
    if stage != "original":
        replan = await _request_replan(db_session, service, goal, run, artifact)
        replacement = await db_session.get(Task, replan.target_id)
        if stage == "accepted_replan":
            candidate = await ingest(
                db_session, goal, run, replacement, None, run.plan_state["planning_version"],
                json.dumps({"schema_version": 1, "plan_kind": "roadmap", "summary": "Revised roadmap",
                            "plan_items": [{**_roadmap_task("task-2"), "why_runnable": "Unstarted work."}]}),
            )
            await _accept_replan_after_selected_authority(
                db_session, service, goal, run, candidate, f"spr208:roadmap:{run.id}:second",
            )
            accepted_version = run.plan_state["planning_version"]
    warnings = [await _warn(service, db_session, run, "Accepted roadmap exists.",
                           {"condition": "plan_missing", "subject": subject}, f"roadmap_{index}")
                for index, subject in enumerate((str(run.id), f"{run.id}:{accepted_version}"))]
    if stage == "pending_replan":
        pending = await _warn(service, db_session, run, "Replacement roadmap missing.",
                              {"condition": "plan_missing", "subject": f"{run.id}:{run.plan_state['planning_version']}"},
                              "replacement_missing")
    await service._reconcile_conditions(db_session, run)
    for warning in warnings:
        await db_session.refresh(warning)
        assert not warning.active, stage
    if stage == "pending_replan":
        await db_session.refresh(pending)
        assert pending.active


async def test_spr208_tick_keeps_warning_when_accepted_snapshot_is_missing(db_session, test_project):
    service, _goal, run, _task, _artifact, _acceptance = await _accepted_plan(db_session, test_project.id)
    warning = await _warn(service, db_session, run, "Missing accepted plan.",
                          {"condition": "plan_missing", "subject": f"{run.id}:1"}, "plan_missing")
    run.plan_state = {k: v for k, v in run.plan_state.items() if k != "accepted_plan_snapshot"}
    run.phase = "authorized"
    await db_session.flush()
    await db_session.commit()  # The missing accepted snapshot is committed fixture state.

    await service.tick(db_session, run.id)

    await db_session.refresh(warning)
    assert warning.active and warning.resolved_at is None


@pytest.mark.parametrize("cursor", [None, {"timestamp": "not-a-date", "id": "not-a-uuid"}])
async def test_spr208_cursor_is_bounded_persistent_and_wraps_after_deleted_tied_row(
    db_session, test_project, monkeypatch, cursor,
):
    service, _goal, run, _agent, _task = await _setup(db_session, test_project)
    gates = [OrchestrationGate(run_id=run.id, success_criterion_key=f"cursor-{index}",
                               gate_type="work_completed", status="open" if index < 2 else "accepted")
             for index in range(4)]
    db_session.add_all(gates)
    await db_session.flush()
    warnings, questions = [], []
    tied = _utcnow() - timedelta(days=1)
    for index in range(52):
        warning = await _warn(service, db_session, run, f"Gate warning {index}",
                              {"condition": "gate_unverified", "subject": str(gates[0].id)}, f"cursor-{index}")
        question = await _ask(service, db_session, run, f"Gate question {index}",
                              {"condition": "gate_unverified", "subject": str(gates[1].id)})
        warning.created_at, question.asked_at = tied, tied
        warnings.append(warning)
        questions.append(question)
    # The last UUID at the shared timestamp is beyond the first bounded page.
    tails = [max(rows, key=lambda row: row.id) for rows in (warnings, questions)]
    for tail, gate in zip(tails, gates[2:]):
        origin = await db_session.scalar(select(OrchestrationAction).where(OrchestrationAction.target_id == tail.id))
        origin.request = {**origin.request, "condition": {"condition": "gate_unverified", "subject": str(gate.id)}}
    run.supervision_state = {"preserved": {"key": "value"}, "condition_reconciliation": {
        "warnings": cursor, "questions": cursor,
    }}
    await db_session.flush()
    calls = {"warnings": 0, "questions": 0}
    original = service._condition_holds

    async def observed(db, current_run, raw):
        kind = "warnings" if raw["subject"] in {str(gates[0].id), str(gates[2].id)} else "questions"
        calls[kind] += 1
        return await original(db, current_run, raw)

    monkeypatch.setattr(service, "_condition_holds", observed)
    assert await service._reconcile_conditions(db_session, run) == 0
    assert calls == {"warnings": 50, "questions": 50}
    await db_session.flush()
    await db_session.refresh(run)
    for kind, model in (("warnings", OrchestrationWarning), ("questions", OrchestrationAuthorityDecision)):
        prior = await db_session.get(model, uuid.UUID(run.supervision_state["condition_reconciliation"][kind]["id"]))
        await db_session.delete(prior)
    await db_session.flush()
    calls.update(warnings=0, questions=0)
    assert await service._reconcile_conditions(db_session, run) == 2
    assert calls == {"warnings": 2, "questions": 2}
    for tail in tails:
        await db_session.refresh(tail)
    assert not tails[0].active and tails[1].status == "cancelled"

    # The end-of-range cursor must wrap and revisit formerly unsatisfied rows.
    gates[0].status = gates[1].status = "accepted"
    calls.update(warnings=0, questions=0)
    assert await service._reconcile_conditions(db_session, run) == 100
    assert calls == {"warnings": 50, "questions": 50}
    await db_session.refresh(run)
    assert run.supervision_state["preserved"] == {"key": "value"}
