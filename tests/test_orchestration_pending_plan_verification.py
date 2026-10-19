"""Pre-acceptance verification uses ingested planning provenance without gate evidence."""
import json
import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from huddleroom.models.orchestration import OrchestrationAction, OrchestrationEvidence
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
from huddleroom.models.artifact import Artifact
from huddleroom.models.task import Task
from huddleroom.models.session import Session
from tests.test_orchestration_plan_ingestion import _planning_run, _env
from tests.test_orchestration_plan_delegation_gate import _agent, _single_plan_gate
from tests.test_orchestration_runtime_e2e import _complete_task_session


async def _pending(db, project_id):
    service, goal, run, task, session = await _planning_run(
        db, project_id, roadmap_budget={"caps": {"max_turns": "30"}},
    )
    task.status = "done"
    run.phase = "authorized"
    item = {"item_key": "audit", "unit_type": "task", "title": "Review local fixtures",
            "work_function": "investigation", "scope": "Read synthetic fixtures", "deliverable": "Review",
            "why_runnable": "Fixtures available"}
    session.output = json.dumps(_env("roadmap", [item]))
    await service._ingest_planning_output(db, run, task, session.id)
    artifact = await db.get(Artifact, uuid.UUID(run.plan_state["plan_artifact_id"]))
    gate = await _single_plan_gate(db, run.id)
    assert gate.required_evidence == {"required_source_types": ["artifact"], "min_count": 1}
    assert run.plan_state["status"] == "requested"
    assert not list(await db.scalars(select(OrchestrationEvidence).where(OrchestrationEvidence.gate_id == gate.id)))
    return service, goal, run, task, session, artifact, gate


@pytest.mark.asyncio
async def test_pending_ingested_plan_verification_roundtrip_and_replay(db_session, test_project):
    service, _, run, producer_task, _, artifact, gate = await _pending(db_session, test_project.id)
    verifier = _agent("independent-reviewer", role="validator", capabilities=["validation"])
    db_session.add(verifier)
    await db_session.flush()
    request = {"action_type": "request_verification", "gate_id": str(gate.id), "work_function": "validation"}
    key = f"pending-plan:{run.id}"
    action = await service.execute_request_verification_action(db_session, run.id, request, key)
    assert action.status == "completed"
    assert action.request["source_task_id"] == str(producer_task.id)
    assert action.request["producer_agent_id"] == str(producer_task.assigned_to)
    verification_task = await db_session.get(Task, action.target_id)
    assert verification_task.assigned_to == verifier.id != artifact.created_by_agent
    assert (await service.execute_request_verification_action(db_session, run.id, request, key)).id == action.id
    assert not list(await db_session.scalars(select(OrchestrationEvidence).where(OrchestrationEvidence.gate_id == gate.id)))
    session = await _complete_task_session(db_session, test_project.id, verification_task, verifier.id,
        '{"status":"done","verdict":"accepted","evidence":["Reviewed saved draft"]}')
    events = await service._new_events(db_session, test_project.id, run.event_cursor)
    await service._ingest_evidence_from_events(db_session, run, events)
    evidence = list(await db_session.scalars(select(OrchestrationEvidence).where(OrchestrationEvidence.gate_id == gate.id)))
    assert any(row.source_type == "session" and row.source_id == session.id
               and row.producer_agent_id == verifier.id for row in evidence)
    assert not any(row.source_type == "artifact" for row in evidence)
    assert run.plan_state["status"] == "requested"


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["run", "version", "task", "session_task", "session_author", "session_status", "author", "gate", "nonwinning_session", "marker_missing", "marker_failed", "marker_target"])
async def test_pending_plan_verification_rejects_invalid_lineage(db_session, test_project, invalid):
    service, _, run, task, session, artifact, gate = await _pending(db_session, test_project.id)
    if invalid == "run":
        task.metadata_ = {**task.metadata_, "orchestration": {**task.metadata_["orchestration"], "run_id": str(uuid.uuid4())}}
    elif invalid in {"version", "task", "author"}:
        field = {"version": "planning_version", "task": "planning_task_id", "author": "plan_author_agent_id"}[invalid]
        artifact.metadata_ = {**artifact.metadata_, field: 99 if invalid == "version" else str(uuid.uuid4())}
    elif invalid == "session_task":
        session.task_id = None
    elif invalid == "session_author":
        other = _agent("foreign-author")
        db_session.add(other)
        await db_session.flush()
        session.agent_id = other.id
    elif invalid == "session_status":
        session.status = "failed"
    elif invalid == "gate":
        run.plan_state = {**run.plan_state, "plan_gate_id": str(uuid.uuid4())}
    elif invalid == "nonwinning_session":
        alternate = Session(task_id=task.id, agent_id=task.assigned_to, project_id=test_project.id,
                            adapter_type="api", status="completed")
        db_session.add(alternate)
        await db_session.flush()
        artifact.metadata_ = {**artifact.metadata_, "winning_session_id": str(alternate.id)}
    elif invalid.startswith("marker_"):
        marker = await db_session.scalar(select(OrchestrationAction).where(
            OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == "report_consumed"))
        if invalid == "marker_missing":
            await db_session.delete(marker)
        elif invalid == "marker_failed":
            marker.status = "failed"
        else:
            marker.target_id = uuid.uuid4()
    await db_session.flush()
    with pytest.raises(HTTPException) as exc:
        await service.execute_request_verification_action(db_session, run.id,
            {"gate_id": str(gate.id), "work_function": "validation"}, f"invalid:{invalid}:{run.id}")
    assert exc.value.status_code == 409
    assert exc.value.detail != "No strong verification agent fit"
    assert not list(await db_session.scalars(select(OrchestrationEvidence).where(OrchestrationEvidence.gate_id == gate.id)))


@pytest.mark.asyncio
async def test_pending_plan_verification_excludes_producer_from_verifier_selection(db_session, test_project):
    service, _, run, task, _, _, gate = await _pending(db_session, test_project.id)
    from huddleroom.models.agent import Agent
    producer = await db_session.get(Agent, task.assigned_to)
    producer.role, producer.capabilities = "validator", ["planning", "validation"]
    await db_session.flush()
    with pytest.raises(HTTPException) as exc:
        await service.execute_request_verification_action(db_session, run.id,
            {"gate_id": str(gate.id), "work_function": "validation"}, f"self:{run.id}")
    assert exc.value.status_code == 409
    assert exc.value.detail == "No strong verification agent fit"


@pytest.mark.asyncio
async def test_manager_decision_task_is_bound_before_planning_only_admission(db_session, test_project):
    """The real authority-delivery flow creates the only manager planning exception."""
    from huddleroom.services.orchestration_authority_service import effective_authorization
    from huddleroom.services.orchestration_plan_review import PlanReviewService
    from huddleroom.services.orchestration_roadmap_service import OrchestrationRoadmapService
    from huddleroom.services.orchestration_authority_service import compute_goal_contract_fingerprint

    service, goal, run, _, _, artifact, _ = await _pending(db_session, test_project.id)
    # This delivery task is intentionally unbudgeted here; the test exercises
    # the authority lineage rather than a separate capped-delegation policy.
    goal.budget = {}
    run.budget_state = {}
    manager = _agent("current-manager", role="manager", capabilities=["management", "planning"])
    db_session.add(manager)
    await db_session.flush()
    goal.manager_user_id = None
    goal.manager_agent_id = manager.id
    review = await PlanReviewService().check(db_session, goal, run, artifact, boundary="manager test")
    assert review.state == "ready"
    binding = OrchestrationRoadmapService._review_binding(review)
    decision = OrchestrationAuthorityDecision(
        goal_id=goal.id,
        run_id=run.id,
        decision_key="roadmap_plan:manager-test",
        title="Manager plan review",
        question="Review the current plan candidate.",
        authority="manager",
        authority_agent_id=manager.id,
        options=[{"key": "approve"}],
        context=json.dumps(binding),
    )
    db_session.add(decision)
    fingerprint = compute_goal_contract_fingerprint(goal)
    run.goal_contract_fingerprint = fingerprint
    run.plan_state = {**run.plan_state, "scope_contract_version": 1}
    db_session.add(OrchestrationAction(
        run_id=run.id,
        idempotency_key=f"run:{run.id}:kind:authorize_execution",
        action_type="authorize_execution",
        status="completed",
        request={"scope": "planning_only", "contract_fingerprint": fingerprint},
        target_type="run",
        target_id=run.id,
    ))
    await db_session.flush()

    summary = await service._sync_agent_authority_decisions(db_session, goal, run)
    await db_session.refresh(decision)
    assert summary == {
        "delegated": 1, "recorded": 0, "escalated": 0,
    }, decision.reason
    action = await db_session.get(OrchestrationAction, decision.related_action_id)
    task = await db_session.get(Task, action.target_id)
    assert action.status == "completed"
    assert action.request["authority_decision_id"] == str(decision.id)
    assert action.request["plan_review_identity_hash"] == binding["plan_review_identity_hash"]
    assert task.metadata_["orchestration"]["authority_decision_id"] == str(decision.id)
    assert task.metadata_["orchestration"]["plan_review_identity_hash"] == binding["plan_review_identity_hash"]
    assert (await effective_authorization(db_session, goal, run, task)).allowed


@pytest.mark.asyncio
async def test_verifier_verdict_correction_keeps_current_planning_only_authority(db_session, test_project):
    """A server-dispatched correction inherits only its live verifier parent's authority."""
    from huddleroom.services.orchestration_authority_service import (
        compute_goal_contract_fingerprint,
        effective_authorization,
    )
    from huddleroom.services.orchestration_result_contract import VERIFICATION_VERDICT

    from huddleroom.services.orchestration_plan_review import PlanReviewService
    from huddleroom.services.orchestration_roadmap_service import OrchestrationRoadmapService

    service, goal, run, planning_task, _, artifact, gate = await _pending(db_session, test_project.id)
    verifier = _agent("independent-verifier", role="validator", capabilities=["validation"])
    db_session.add(verifier)
    await db_session.flush()
    review = await PlanReviewService().check(db_session, goal, run, artifact, boundary="verifier correction test")
    binding = OrchestrationRoadmapService._review_binding(review)
    gate.gate_type = "roadmap_plan_approval"
    gate.required_evidence = {
        **gate.required_evidence,
        "plan_artifact_id": str(artifact.id),
        "planning_task_id": str(planning_task.id),
        "work_producer_agent_id": str(planning_task.assigned_to),
        **binding,
    }
    action = await service.execute_request_verification_action(
            db_session, run.id,
            {"action_type": "request_verification", "gate_id": str(gate.id), "work_function": "validation"},
            f"verifier-correction:{run.id}",
    )
    parent = await db_session.get(Task, action.target_id)
    assert parent.assigned_to == verifier.id
    fingerprint = compute_goal_contract_fingerprint(goal)
    run.goal_contract_fingerprint = fingerprint
    run.plan_state = {**run.plan_state, "scope_contract_version": 1}
    db_session.add(OrchestrationAction(run_id=run.id, idempotency_key=f"run:{run.id}:kind:authorize_execution",
        action_type="authorize_execution", status="completed", request={"scope": "planning_only", "contract_fingerprint": fingerprint}, target_type="run", target_id=run.id))
    ledger = {"kind": VERIFICATION_VERDICT, "cycle": 1, "attempts": [], "status": "pending"}
    await service._dispatch_result_correction(
            db_session, goal, run, parent, ledger, "invalid verdict",
            {"task_id": str(parent.id), "session_id": None, "errors": ["invalid verdict"]},
            {}, f"task:{parent.id}:result_correction", [],
            service._result_correction_version(goal, run),
    )
    assert "correction_task_id" in ledger, ledger.get("detail")
    correction = await db_session.get(Task, uuid.UUID(ledger["correction_task_id"]))
    correction_action = await db_session.get(OrchestrationAction, uuid.UUID(correction.metadata_["orchestration"]["action_id"]))
    assert correction_action.status == "completed"
    assert correction.metadata_["result_correction_parent_task_id"] == str(parent.id)
    assert (await effective_authorization(db_session, goal, run, correction)).allowed


@pytest.mark.asyncio
async def test_pending_current_artifact_overrides_prior_task_evidence(db_session, test_project):
    service, _, run, task, _, _, gate = await _pending(db_session, test_project.id)
    verifier = _agent("current-verifier", role="validator", capabilities=["validation"])
    prior_author = _agent("prior-planner")
    db_session.add_all([verifier, prior_author])
    await db_session.flush()
    from huddleroom.schemas.task import TaskCreate
    from huddleroom.services.task_service import TaskService
    prior = await TaskService().create(db_session, test_project.id, TaskCreate(title="Prior planner", assigned_to=prior_author.id))
    db_session.add(OrchestrationEvidence(run_id=run.id, gate_id=gate.id, source_type="task", source_id=prior.id,
                                       producer_agent_id=prior_author.id, verdict="candidate", evidence_metadata={}))
    await db_session.flush()
    action = await service.execute_request_verification_action(db_session, run.id,
        {"gate_id": str(gate.id), "work_function": "validation"}, f"prior:{run.id}")
    assert action.request["source_task_id"] == str(task.id)
    assert action.request["producer_agent_id"] == str(task.assigned_to)


def test_saved_draft_48_turn_allocation_still_exceeds_parent_cap():
    from types import SimpleNamespace
    from huddleroom.services.orchestration_budget_service import validate_declared_allocations
    from huddleroom.services.orchestration_roadmap_service import parse_roadmap_items
    items = [{"item_key": f"child_{index}", "unit_type": "goal", "title": "Synthetic child",
              "objective": "Deliver local mock work", "success_criteria": [{"key": "done"}],
              "why_runnable": "Synthetic fixtures available", "allocation": {"max_turns": "8"}}
             for index in range(6)]
    with pytest.raises(HTTPException, match="Roadmap allocations exceed parent cap 'max_turns'"):
        validate_declared_allocations(SimpleNamespace(budget={"caps": {"max_turns": "30"}}),
                                      parse_roadmap_items(items, {"max_turns"}))
