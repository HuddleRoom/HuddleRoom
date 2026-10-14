"""Pre-acceptance verification uses ingested planning provenance without gate evidence."""
import json
import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from huddleroom.models.orchestration import OrchestrationAction, OrchestrationEvidence
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
