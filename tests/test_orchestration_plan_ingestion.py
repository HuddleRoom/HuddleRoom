import json
import uuid
from types import SimpleNamespace

import pytest
import pytest_asyncio
from fastapi import HTTPException
from sqlalchemy import func, select

from huddleroom.models.agent import Agent
from huddleroom.models.artifact import Artifact
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.services.orchestration_plan_ingestion import (
    PlanIngestionError,
    check_disposition,
    ingest,
    ingested_plan_hash,
    parse_envelope,
    plan_artifact_id,
    plan_output_hash,
)
from huddleroom.services.orchestration_plan_review import render_markdown

GOAL = SimpleNamespace(success_criteria=[{"key": "c1"}], budget={"caps": {"max_tokens": "1000"}})


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [{}, {"caps": {}}])
async def test_roadmap_missing_budget_requires_attention(budget_session, test_project, test_user, monkeypatch, budget):
    db_session = budget_session
    from copy import deepcopy
    from huddleroom.models.orchestration import (
        OrchestrationAction, OrchestrationGate, OrchestrationGoal, OrchestrationBudgetReservation,
    )
    from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService

    service, goal, run, task, session = await _planning_run(db_session, test_project.id, roadmap_budget=budget)
    original_budget, original_constraints = deepcopy(goal.budget), deepcopy(goal.constraints)
    from huddleroom.schemas.task import TaskCreate
    from huddleroom.services.task_service import TaskService
    audit = await TaskService().create(db_session, test_project.id, TaskCreate(
        title="Existing audit", status="done", metadata={"orchestration": {
            "goal_id": str(goal.id), "run_id": str(run.id), "work_function": "investigation",
        }},
    ))
    audit_id, audit_status = audit.id, audit.status
    child = {"item_key": "child", "unit_type": "goal", "title": "Child", "objective": "Deliver child",
             "success_criteria": [{"key": "done"}], "allocation": {}, "why_runnable": "ready",
             "scope": "invalid task field"}
    malformed_task = {"item_key": "audit", "unit_type": "task", "title": "Audit",
                      "work_function": "investigation", "scope": "Inspect", "deliverable": "Audit",
                      "why_runnable": "ready", "staging_boundary": {"type": "fictional", "identifier": "x"}}
    output = json.dumps(_env("roadmap", [child, malformed_task]))
    await _finish(db_session, task, session, output)
    before = {model: await db_session.scalar(select(func.count()).select_from(model))
              for model in (Task, Artifact, OrchestrationGoal, OrchestrationGate, OrchestrationBudgetReservation)}
    provider_calls = []

    async def no_provider(*args, **kwargs):
        provider_calls.append((args, kwargs))
        pytest.fail("missing owner budget must not invoke a provider")

    monkeypatch.setattr(service, "request_llm_decision", no_provider)
    await service._ingest_session_evidence(db_session, run, _completed(session.id))
    assert run.plan_state["status"] == "budget_required"
    assert run.plan_state["budget_required"]["code"] == "roadmap_child_budget_required"
    assert run.plan_state["budget_required"]["raw_output"] == output
    diagnostics = json.dumps(run.plan_state["budget_required"]["diagnostics"])
    assert "allocation" in diagnostics and "staging boundary" in diagnostics
    assert run.status == "blocked" and "plan_artifact_id" not in run.plan_state
    questions = list(await db_session.scalars(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == "ask_human",
    )))
    assert len(questions) == 1
    assert "Create a new goal with limits" in json.dumps(questions[0].request)

    async def assert_blocked_replay():
        await service._ingest_session_evidence(db_session, run, _completed(session.id))
        await db_session.flush()
        await db_session.commit()
        await service.tick(db_session, run.id, local_only=True)
        with pytest.raises(HTTPException) as exc:
            await service.execute_request_plan_revision_action(
                db_session, run_id=run.id,
                request={"action_type": "request_plan_revision", "plan_task_id": str(task.id),
                         "revision_request": "Try again"},
                idempotency_key=f"run:{run.id}:kind:request_plan_revision:budget-test",
            )
        assert exc.value.status_code == 409 and exc.value.detail == "roadmap_child_budget_required"

    await assert_blocked_replay()
    decision = await db_session.get(OrchestrationAuthorityDecision, questions[0].target_id)
    assert decision.title == "Budget required"
    await OrchestrationAuthorityDecisionService().answer_runtime_question(
        db_session, decision, "acknowledge", actor_user_id=test_user.id,
        contract_version=decision.contract_version, reason="Keep blocked until a new goal has limits",
    )
    await assert_blocked_replay()
    assert run.status == "blocked" and not provider_calls
    assert goal.budget == original_budget and goal.constraints == original_constraints
    assert session.output == output and run.plan_state["planning_task_id"] == str(task.id)
    assert (await db_session.get(Task, audit_id)).status == audit_status
    for model, count in before.items():
        assert await db_session.scalar(select(func.count()).select_from(model)) == count
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == "ask_human",
    )) == 1


def test_task_only_roadmap_needs_no_parent_caps():
    item = {"item_key": "task", "unit_type": "task", "title": "Task", "work_function": "investigation",
            "scope": "Inspect", "deliverable": "Audit", "why_runnable": "ready"}
    goal = SimpleNamespace(success_criteria=[{"key": "c1"}], budget={})
    assert _parse(_env("roadmap", [item]), "roadmap", goal)["plan_items"][0]["item_key"] == "task"


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


async def _planning_run(db_session, project_id, *, roadmap_budget=None):
    from tests.test_orchestration_plan_delegation_gate import _agent, _make_run, _plan_request

    planner = _agent("planner")
    db_session.add(planner)
    await db_session.flush()
    if roadmap_budget is None:
        service, goal, run = await _make_run(db_session, project_id)
    else:
        from huddleroom.schemas.orchestration import OrchestrationGoalCreate
        from huddleroom.services.orchestration_service import OrchestrationService
        from tests.test_orchestration_plan_delegation_gate import _complete_goal_definition

        service = OrchestrationService()
        goal, run = await service.create_goal(
            db_session, project_id=project_id, created_by_user_id=None,
            data=OrchestrationGoalCreate(objective="Deliver synthetic roadmap", goal_type="roadmap",
                                         success_criteria=[{"key": "plan"}], budget=roadmap_budget),
        )
        await _complete_goal_definition(db_session, goal, run)
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
    from huddleroom.models.agent import Agent
    from tests.conftest import bind_plan_author_team

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

    await bind_plan_author_team(db_session, goal, await db_session.get(Agent, task.assigned_to))
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


@pytest.mark.asyncio
async def test_acceptance_binds_ingested_artifact_even_after_metadata_rewrite_drops_source(db_session, test_project):
    service, goal, run, task, session = await _planning_run(db_session, test_project.id)
    artifact = await ingest(db_session, goal, run, task, session.id, run.plan_state["planning_version"], _db_output())

    # PUT /artifacts replaces metadata wholesale; the server-held plan_artifact_id still binds the content.
    artifact.metadata_ = {"plan_items": [_item(id="b", success_criterion_keys=["plan"])]}
    await db_session.flush()
    with pytest.raises(HTTPException) as exc:
        await service._plan_artifact_for_run(db_session, run, artifact.id)
    assert exc.value.status_code == 409


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

    # A replayed task event after completed ingestion is a no-op.
    assert await service._ingest_task_evidence(db_session, run, task_event) == 0
    assert await _plan_artifacts(db_session, task.id) == 1
    assert run.plan_state == state


@pytest.mark.asyncio
async def test_gated_planning_work_item_is_not_ingested_as_plan_output(db_session, test_project):
    from huddleroom.models.event_log import EventLog
    from huddleroom.models.orchestration import OrchestrationEvidence, OrchestrationGate
    from huddleroom.schemas.task import TaskCreate
    from huddleroom.services.event_bus import emit_event_once
    from huddleroom.services.task_service import TaskService

    service, goal, run, _, _ = await _planning_run(db_session, test_project.id)
    gate = OrchestrationGate(run_id=run.id, success_criterion_key="plan_item:p", gate_type="work_completed", required_evidence={})
    db_session.add(gate)
    await db_session.flush()
    item_task = await TaskService().create(db_session, test_project.id, TaskCreate(
        title="Planning work item", metadata={"orchestration": {
            "goal_id": str(goal.id), "run_id": str(run.id), "work_function": "planning", "gate_id": str(gate.id),
        }},
    ))
    session = Session(task_id=item_task.id, agent_id=(await db_session.get(Task, uuid.UUID(run.plan_state["planning_task_id"]))).assigned_to,
                      project_id=test_project.id, adapter_type="api")
    db_session.add(session)
    await _finish(db_session, item_task, session, _db_output())
    await emit_event_once(
        db_session, test_project.id, "task.status_changed", {"task_id": str(item_task.id), "status": "done"},
        dedup_key=f"gated-planning-item:{item_task.id}",
    )
    event = await db_session.scalar(
        select(EventLog).where(EventLog.event_type == "task.status_changed").order_by(EventLog.seq.desc()).limit(1)
    )
    state = dict(run.plan_state)

    assert await service._ingest_task_evidence(db_session, run, event) == 1

    assert await _plan_artifacts(db_session, item_task.id) == 0 and run.plan_state == state
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationEvidence).where(
        OrchestrationEvidence.gate_id == gate.id
    )) == 1


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
async def test_ingested_plan_artifact_survives_later_prose_output(db_session, test_project):
    from huddleroom.models.agent import Agent
    from tests.conftest import bind_plan_author_team

    service, goal, run, task, session = await _planning_run(db_session, test_project.id)
    artifact = await ingest(
        db_session, goal, run, task, session.id, run.plan_state["planning_version"], _db_output(),
    )
    await bind_plan_author_team(db_session, goal, await db_session.get(Agent, task.assigned_to))
    await _finish(db_session, task, session, "I registered the plan artifact.")

    await service._ingest_session_evidence(db_session, run, _completed(session.id))

    assert run.plan_state["status"] == "requested"
    action = await service.execute_accept_plan_action(
        db_session, run_id=run.id, request={"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
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


async def _race_ingest(db_session, test_project, row_summary):
    """Row for (task, version) lands between the existence check and the insert."""
    from sqlalchemy import insert

    _, goal, run, task, session = await _planning_run(db_session, test_project.id)
    version = run.plan_state["planning_version"]
    normalized = parse_envelope(_db_output(row_summary), "implementation", goal)
    aid = plan_artifact_id(task.id, version)
    await db_session.execute(insert(Artifact.__table__).values(
        id=aid, project_id=test_project.id, name="raced", artifact_type="plan", status="draft",
        linked_task_id=task.id, created_by_agent=task.assigned_to,
        content_hash=plan_output_hash(normalized), metadata={"output_hash": plan_output_hash(normalized)},
    ))
    real_get, calls = db_session.get, []

    async def get(model, ident, *a, **kw):
        if model is Artifact:
            calls.append(ident)
            if len(calls) == 1:
                return None  # simulate: row not yet visible at check time
        return await real_get(model, ident, *a, **kw)

    db_session.get = get
    try:
        return await ingest(db_session, goal, run, task, session.id, version, _db_output()), run, task
    finally:
        del db_session.get


@pytest.mark.asyncio
async def test_ingest_integrity_race_same_hash_returns_existing(db_session, test_project):
    artifact, run, task = await _race_ingest(db_session, test_project, "sum")
    assert artifact.id == plan_artifact_id(task.id, run.plan_state["planning_version"])
    assert run.plan_state["plan_artifact_id"] == str(artifact.id)
    assert await _plan_artifacts(db_session, task.id) == 1


@pytest.mark.asyncio
async def test_ingest_integrity_race_different_hash_conflicts(db_session, test_project):
    with pytest.raises(PlanIngestionError, match="conflicting plan output"):
        await _race_ingest(db_session, test_project, "other")


@pytest.mark.asyncio
async def test_ingest_rejects_invalid_adoption_before_disposition(db_session, test_project):
    from huddleroom.schemas.task import TaskCreate
    from huddleroom.services.task_service import TaskService

    _, goal, run, task, session = await _planning_run(db_session, test_project.id)
    audit = await TaskService().create(db_session, test_project.id, TaskCreate(
        title="Audit",
        metadata={
            "orchestration": {"goal_id": str(goal.id), "run_id": str(run.id), "work_function": "investigation"},
            "orchestration_contract": {"work_function": "investigation"},
        },
    ))
    plan_meta = task.metadata_["orchestration_plan"]
    task.metadata_ = {**task.metadata_, "orchestration_plan": {**plan_meta, "pre_plan_task_ids": [str(audit.id)]}}
    output = json.dumps(_env(items=[_item(success_criterion_keys=["plan"], adopt_task_id=str(audit.id))]))

    with pytest.raises(PlanIngestionError) as exc:
        await ingest(db_session, goal, run, task, session.id, run.plan_state["planning_version"], output)

    # Contract metadata alone is not enough: no delegation action created this task.
    assert exc.value.reason == f"adoption_invalid: item a: task {audit.id} was not created by a delegation"
    assert not exc.value.stale
    assert await _plan_artifacts(db_session, task.id) == 0


@pytest.mark.asyncio
async def test_ingest_rejects_plan_item_artifact_outside_project(db_session, test_project):
    _, goal, run, task, session = await _planning_run(db_session, test_project.id)
    ghost = uuid.uuid4()
    output = json.dumps(_env(items=[_item(success_criterion_keys=["plan"], artifact_ids=[str(ghost)])]))

    with pytest.raises(PlanIngestionError) as exc:
        await ingest(db_session, goal, run, task, session.id, run.plan_state["planning_version"], output)

    assert exc.value.reason == f"artifact_invalid: item a: artifact {ghost} not found in project"
    assert not exc.value.stale
    assert await _plan_artifacts(db_session, task.id) == 0


@pytest.mark.asyncio
async def test_ingest_accepts_plan_item_artifact_in_project(db_session, test_project):
    _, goal, run, task, session = await _planning_run(db_session, test_project.id)
    known = Artifact(project_id=test_project.id, name="spec", artifact_type="document")
    db_session.add(known)
    await db_session.flush()
    output = json.dumps(_env(items=[_item(success_criterion_keys=["plan"], artifact_ids=[str(known.id)])]))

    await ingest(db_session, goal, run, task, session.id, run.plan_state["planning_version"], output)

    assert await _plan_artifacts(db_session, task.id) == 1


@pytest_asyncio.fixture
async def budget_session(db_session, test_project, test_user, test_engine):
    """Publish only budget-case fixture actors, then use a real independent session."""
    from sqlalchemy.ext.asyncio import AsyncSession
    await db_session.flush()
    await db_session.commit()
    async with AsyncSession(test_engine, expire_on_commit=False) as independent:
        yield independent
        await independent.rollback()


async def _accepted_team(db_session, goal, agent):
    from tests.conftest import bind_plan_author_team

    await bind_plan_author_team(db_session, goal, agent, role_agents={"implementation": agent})


@pytest.mark.asyncio
async def test_ingest_rejects_work_function_outside_accepted_team(db_session, test_project):
    service, goal, run, task, session = await _planning_run(db_session, test_project.id)
    await _accepted_team(db_session, goal, await db_session.get(Agent, task.assigned_to))
    version = run.plan_state["planning_version"]
    output = json.dumps(_env(items=[_item(work_function="documentation", success_criterion_keys=["plan"])]))

    with pytest.raises(PlanIngestionError) as exc:
        await ingest(db_session, goal, run, task, session.id, version, output)

    assert exc.value.correctable and not exc.value.stale
    [diagnostic] = exc.value.diagnostics
    assert "documentation" in diagnostic and "[implementation]" in diagnostic
    assert await _plan_artifacts(db_session, task.id) == 0
    assert await ingest(db_session, goal, run, task, session.id, version, _db_output()) is not None


@pytest.mark.asyncio
async def test_ingest_rejects_foreign_agent_for_accepted_function(db_session, test_project):
    from tests.test_orchestration_plan_delegation_gate import _agent

    _, goal, run, task, session = await _planning_run(db_session, test_project.id)
    await _accepted_team(db_session, goal, await db_session.get(Agent, task.assigned_to))
    foreign = _agent("intruder")
    db_session.add(foreign)
    await db_session.flush()
    output = json.dumps(_env(items=[_item(agent_id=str(foreign.id), success_criterion_keys=["plan"])]))

    with pytest.raises(PlanIngestionError) as exc:
        await ingest(db_session, goal, run, task, session.id, run.plan_state["planning_version"], output)

    assert exc.value.correctable
    assert exc.value.diagnostics == ["item 'a': agent_id is outside the accepted team for 'implementation'"]


@pytest.mark.asyncio
async def test_ingest_skips_human_and_goal_items_in_team_check(db_session, test_project):
    _, goal, run, task, session = await _planning_run(db_session, test_project.id)
    await _accepted_team(db_session, goal, await db_session.get(Agent, task.assigned_to))
    human = _item(id="consent", owner_type="human", work_function=None, success_criterion_keys=["plan"])
    output = json.dumps(_env(items=[human, _item(success_criterion_keys=["plan"])]))

    artifact = await ingest(db_session, goal, run, task, session.id, run.plan_state["planning_version"], output)

    assert artifact.metadata_["plan_items"][0]["owner_type"] == "human"


def test_parse_envelope_makes_omitted_task_owner_type_explicit_agent():
    out = _parse(_env(items=[_item(), _item(id="human", owner_type="human", work_function=None)]))
    assert [item["owner_type"] for item in out["plan_items"]] == ["agent", "human"]


def test_roadmap_parse_makes_omitted_task_owner_type_explicit_agent():
    item = {"item_key": "task", "unit_type": "task", "title": "Task", "work_function": "investigation",
            "scope": "Inspect", "deliverable": "Audit", "why_runnable": "ready"}
    assert _parse(_env("roadmap", [item]), "roadmap")["plan_items"][0]["owner_type"] == "agent"


@pytest.mark.asyncio
async def test_ingest_stores_explicit_owner_type_and_renders_no_null(db_session, test_project):
    _, goal, run, task, session = await _planning_run(db_session, test_project.id)
    await _accepted_team(db_session, goal, await db_session.get(Agent, task.assigned_to))
    human = _item(id="consent", owner_type="human", work_function=None, success_criterion_keys=["plan"])
    output = json.dumps(_env(items=[human, _item(success_criterion_keys=["plan"])]))

    artifact = await ingest(db_session, goal, run, task, session.id, run.plan_state["planning_version"], output)

    assert [item["owner_type"] for item in artifact.metadata_["plan_items"]] == ["human", "agent"]
    assert ingested_plan_hash(artifact.metadata_) == artifact.content_hash
    identity = SimpleNamespace(artifact_id=artifact.id, planning_task_id=task.id,
                               planning_version=run.plan_state["planning_version"],
                               canonical_content_hash=artifact.content_hash)
    markdown = render_markdown(identity, artifact.metadata_, {})
    assert "Owner type: null" not in markdown
    assert '- Owner type: "agent"' in markdown and '- Owner type: "human"' in markdown


@pytest.mark.asyncio
async def test_ingest_skips_goal_items_in_team_check(db_session, test_project):
    _, goal, run, task, session = await _planning_run(
        db_session, test_project.id, roadmap_budget={"caps": {"max_tokens": "1000"}},
    )
    await _accepted_team(db_session, goal, await db_session.get(Agent, task.assigned_to))
    child = {"item_key": "child", "unit_type": "goal", "title": "Child", "objective": "Deliver child",
             "success_criteria": [{"key": "done"}], "allocation": {"max_tokens": 1}, "why_runnable": "ready"}

    artifact = await ingest(db_session, goal, run, task, session.id, run.plan_state["planning_version"],
                            json.dumps(_env("roadmap", [child])))

    assert artifact.metadata_["plan_items"][0]["item_key"] == "child"


@pytest.mark.asyncio
async def test_ingest_without_accepted_team_skips_membership_check(db_session, test_project):
    _, goal, run, task, session = await _planning_run(db_session, test_project.id)
    output = json.dumps(_env(items=[_item(work_function="documentation", success_criterion_keys=["plan"])]))

    artifact = await ingest(db_session, goal, run, task, session.id, run.plan_state["planning_version"], output)

    assert artifact.metadata_["plan_items"][0]["work_function"] == "documentation"
