import json
import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import func, select

from huddleroom.models.artifact import Artifact
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.services.orchestration_plan_ingestion import (
    PlanIngestionError,
    check_disposition,
    ingest,
    parse_envelope,
    plan_artifact_id,
    plan_output_hash,
)

GOAL = SimpleNamespace(success_criteria=[{"key": "c1"}], budget={"caps": {"max_tokens": "1000"}})


def _item(**kw):
    base = {"id": "a", "work_function": "implementation", "scope": "s", "deliverable": "d", "why_runnable": "ready"}
    return {**base, **kw}


def _env(kind="implementation", items=None, **kw):
    return {"schema_version": 1, "plan_kind": kind, "summary": "sum", "plan_items": items or [_item()], **kw}


def _parse(data, kind="implementation", goal=GOAL):
    return parse_envelope(data if isinstance(data, str) else json.dumps(data), kind, goal)


def _rejects(data, text, kind="implementation"):
    with pytest.raises(PlanIngestionError) as exc:
        _parse(data, kind)
    assert text in exc.value.reason
    assert not exc.value.stale


@pytest.mark.parametrize("kind", ["implementation", "delivery"])
def test_valid_outcome_envelopes(kind):
    out = _parse(_env(kind), kind)
    assert out["plan_kind"] == kind and out["plan_items"][0]["id"] == "a"


def test_valid_roadmap_envelope():
    item = {
        "item_key": "k", "unit_type": "task", "title": "t", "work_function": "implementation",
        "scope": "s", "deliverable": "d", "why_runnable": "ready", "allocation": {},
    }
    assert _parse(_env("roadmap", [item]), "roadmap")["plan_items"][0]["item_key"] == "k"


def test_rejections():
    _rejects("here is my plan", "JSON object")
    _rejects("```json\n" + json.dumps(_env()) + "\n```", "JSON object")
    _rejects(_env(extra=1), "extra: Extra inputs")
    _rejects(_env(schema_version=2), "schema_version")
    _rejects(_env("delivery"), "plan_kind 'delivery' does not match expected 'implementation'")
    _rejects(_env(items=[_item(why_runnable="  ")]), "why_runnable")
    _rejects('{"schema_version":1,"schema_version":1}', "duplicate key")
    _rejects(_env(items=[_item(depends_on=["zzz"])]), "unknown plan item 'zzz'")
    _rejects(_env(items=[_item(id="a", depends_on=["b"]), _item(id="b", depends_on=["a"])]), "dependency cycle")


def test_disposition():
    tid = uuid.uuid4()
    norm = _parse(_env())
    with pytest.raises(PlanIngestionError, match=str(tid)):
        check_disposition(norm, [tid])
    check_disposition({**norm, "summary": f"skipping {tid}: obsolete"}, [tid])
    adopted = _parse(_env(items=[_item(adopt_task_id=str(tid))]))
    check_disposition(adopted, [tid])


def test_ids_and_hash():
    t = uuid.uuid4()
    assert plan_artifact_id(t, 1) == plan_artifact_id(t, 1) != plan_artifact_id(t, 2)
    assert plan_output_hash({"a": 1, "b": 2}) == plan_output_hash({"b": 2, "a": 1})


async def _planning_run(db_session, project_id):
    from tests.test_orchestration_plan_delegation_gate import _agent, _make_run, _plan_request

    planner = _agent("planner")
    db_session.add(planner)
    await db_session.flush()
    service, goal, run = await _make_run(db_session, project_id)
    action = await service.execute_request_plan_action(
        db_session, run_id=run.id, request=_plan_request(planner.id), idempotency_key=f"run:{run.id}:kind:request_plan"
    )
    task = await db_session.get(Task, action.target_id)
    session = Session(task_id=task.id, agent_id=planner.id, project_id=project_id, adapter_type="api", status="completed")
    db_session.add(session)
    await db_session.flush()
    return service, goal, run, task, session


def _db_output(summary="sum"):
    return json.dumps(_env(items=[_item(success_criterion_keys=["plan"])], summary=summary))


async def _plan_artifacts(db_session, task_id):
    return (await db_session.execute(
        select(func.count()).select_from(Artifact).where(Artifact.linked_task_id == task_id)
    )).scalar_one()


@pytest.mark.asyncio
async def test_ingest_creates_bound_artifact_replays_and_accepts(db_session, test_project):
    service, goal, run, task, session = await _planning_run(db_session, test_project.id)
    version = run.plan_state["planning_version"]

    artifact = await ingest(db_session, goal, run, task, session.id, version, _db_output())

    assert artifact.id == plan_artifact_id(task.id, version)
    assert (artifact.project_id, artifact.artifact_type, artifact.status) == (test_project.id, "plan", "draft")
    assert (artifact.linked_task_id, artifact.created_by_agent) == (task.id, task.assigned_to)
    meta = artifact.metadata_
    assert meta["source"] == "planning_output_ingestion" and meta["kind"] == "implementation_plan"
    assert meta["winning_session_id"] == str(session.id) and meta["planning_version"] == version
    assert artifact.content_hash == meta["output_hash"] == run.plan_state["plan_output_hash"]
    assert run.plan_state["plan_artifact_id"] == str(artifact.id)

    assert (await ingest(db_session, goal, run, task, session.id, version, _db_output())).id == artifact.id
    with pytest.raises(PlanIngestionError, match="conflicting plan output") as exc:
        await ingest(db_session, goal, run, task, session.id, version, _db_output("other"))
    assert not exc.value.stale
    assert await _plan_artifacts(db_session, task.id) == 1

    action = await service.execute_accept_plan_action(
        db_session, run_id=run.id, request={"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
        idempotency_key=f"run:{run.id}:kind:accept_plan",
    )
    assert action.status == "completed" and run.plan_state["status"] == "accepted"


@pytest.mark.asyncio
async def test_ingest_rejects_stale_and_foreign_author(db_session, test_project):
    from tests.test_orchestration_plan_delegation_gate import _agent

    _, goal, run, task, session = await _planning_run(db_session, test_project.id)
    version = run.plan_state["planning_version"]
    with pytest.raises(PlanIngestionError, match="superseded") as exc:
        await ingest(db_session, goal, run, task, session.id, version + 1, _db_output())
    assert exc.value.stale
    run.plan_state = {**run.plan_state, "planning_task_id": str(uuid.uuid4())}
    with pytest.raises(PlanIngestionError, match="not the current") as exc:
        await ingest(db_session, goal, run, task, session.id, version, _db_output())
    assert exc.value.stale
    run.plan_state = {**run.plan_state, "planning_task_id": str(task.id)}

    other = _agent("intruder")
    db_session.add(other)
    await db_session.flush()
    foreign = Session(task_id=task.id, agent_id=other.id, project_id=test_project.id, adapter_type="api")
    db_session.add(foreign)
    await db_session.flush()
    with pytest.raises(PlanIngestionError, match="author") as exc:
        await ingest(db_session, goal, run, task, foreign.id, version, _db_output())
    assert not exc.value.stale
    assert await _plan_artifacts(db_session, task.id) == 0


@pytest.mark.asyncio
async def test_acceptance_rejects_superseded_and_tampered_ingested_artifact(db_session, test_project):
    service, goal, run, task, session = await _planning_run(db_session, test_project.id)
    version = run.plan_state["planning_version"]
    artifact = await ingest(db_session, goal, run, task, session.id, version, _db_output())

    artifact.metadata_ = {**artifact.metadata_, "plan_items": [_item(id="b", success_criterion_keys=["plan"])]}
    await db_session.flush()
    with pytest.raises(HTTPException, match="does not match the ingested"):
        await service._plan_artifact_for_run(db_session, run, artifact.id)

    run.plan_state = {**run.plan_state, "planning_version": version + 1}
    with pytest.raises(HTTPException, match="superseded planning version"):
        await service._plan_artifact_for_run(db_session, run, artifact.id)


def _completed(session_id):
    return SimpleNamespace(id=uuid.uuid4(), event_type="session.completed", payload={"session_id": str(session_id)})


async def _finish(db_session, task, session, output):
    session.output, session.status, task.status = output, "completed", "done"
    await db_session.flush()


@pytest.mark.asyncio
async def test_session_hook_ingests_winner_and_ignores_nonwinning_session(db_session, test_project):
    service, _, run, task, winner = await _planning_run(db_session, test_project.id)
    loser = Session(task_id=task.id, agent_id=task.assigned_to, project_id=test_project.id, adapter_type="api")
    db_session.add(loser)
    await _finish(db_session, task, winner, _db_output())
    loser.output, loser.status = _db_output("other"), "completed"
    await db_session.flush()

    task_event = SimpleNamespace(id=uuid.uuid4(), payload={"status": "done", "task_id": str(task.id)})
    assert await service._ingest_task_evidence(db_session, run, task_event) == 0
    assert await _plan_artifacts(db_session, task.id) == 0  # two completed sessions: no canonical winner yet

    assert await service._ingest_session_evidence(db_session, run, _completed(winner.id)) == 0
    artifact = await db_session.get(Artifact, uuid.UUID(run.plan_state["plan_artifact_id"]))
    assert artifact.metadata_["winning_session_id"] == str(winner.id)
    state = dict(run.plan_state)

    await service._ingest_session_evidence(db_session, run, _completed(loser.id))
    assert await _plan_artifacts(db_session, task.id) == 1
    assert run.plan_state == state


@pytest.mark.asyncio
async def test_invalid_output_requires_revision_and_revision_dispatches_next_version(db_session, test_project):
    service, _, run, task, session = await _planning_run(db_session, test_project.id)
    version = run.plan_state["planning_version"]
    await _finish(db_session, task, session, "Here is my plan: do the work.")

    await service._ingest_session_evidence(db_session, run, _completed(session.id))

    assert run.plan_state["status"] == "revision_required"
    assert run.plan_state["revision_reason"] == "planning_output_rejected: output must be exactly one JSON object"
    assert "plan_artifact_id" not in run.plan_state and await _plan_artifacts(db_session, task.id) == 0
    marker = await service._existing_action_for_key(db_session, run.id, f"run:{run.id}:kind:report_consumed:task:{task.id}")
    assert marker.status == "completed" and marker.request["plan_ingestion_error"].startswith("output must be")
    clarifications = (await db_session.execute(select(Task))).scalars().all()
    assert not [t for t in clarifications if service._task_work_function(t) == "report_clarification"]

    await service.execute_request_plan_revision_action(
        db_session, run_id=run.id,
        request={"action_type": "request_plan_revision", "plan_task_id": str(task.id), "revision_request": "Return JSON."},
        idempotency_key=f"run:{run.id}:kind:request_plan_revision:json",
    )
    revised = await db_session.get(Task, uuid.UUID(run.plan_state["planning_task_id"]))
    assert revised.id != task.id and run.plan_state["status"] == "requested"
    assert revised.metadata_["orchestration_plan"]["planning_version"] == run.plan_state["planning_version"] > version
    inputs = revised.metadata_["orchestration_contract"]["inputs"]
    assert "Plan revision request: Return JSON." in inputs
    assert "Plan revision reason: planning_output_rejected: output must be exactly one JSON object" in inputs


@pytest.mark.asyncio
async def test_manual_plan_artifact_survives_prose_output(db_session, test_project):
    service, _, run, task, session = await _planning_run(db_session, test_project.id)
    manual = Artifact(
        project_id=test_project.id, name="manual-plan", artifact_type="plan", status="draft",
        linked_task_id=task.id, created_by_agent=task.assigned_to,
        metadata_={"plan_items": [_item(success_criterion_keys=["plan"])]},
    )
    db_session.add(manual)
    await _finish(db_session, task, session, "I registered the plan artifact.")

    await service._ingest_session_evidence(db_session, run, _completed(session.id))

    assert run.plan_state["status"] == "requested"
    action = await service.execute_accept_plan_action(
        db_session, run_id=run.id, request={"action_type": "accept_plan", "plan_artifact_id": str(manual.id)},
        idempotency_key=f"run:{run.id}:kind:accept_plan",
    )
    assert action.status == "completed" and run.plan_state["status"] == "accepted"


@pytest.mark.asyncio
async def test_rejected_replan_output_is_surfaced_on_pending_replan(db_session, test_project):
    service, _, run, task, session = await _planning_run(db_session, test_project.id)
    run.plan_state = {**run.plan_state, "status": "accepted", "pending_replan": {"task_id": str(task.id)}}
    await _finish(db_session, task, session, "prose")

    await service._ingest_session_evidence(db_session, run, _completed(session.id))

    assert run.plan_state["status"] == "accepted"
    assert run.plan_state["pending_replan"]["ingestion_error"].startswith("planning_output_rejected: output must be")
