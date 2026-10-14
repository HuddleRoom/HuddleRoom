import json

import pytest
from sqlalchemy import select

from huddleroom.models.base import _utcnow
from huddleroom.models.event_log import EventLog
from huddleroom.models.orchestration import OrchestrationAction
from huddleroom.models.task import Task
from huddleroom.schemas.orchestration import OrchestrationDelegationContract
from huddleroom.services.event_bus import emit_event_once
from huddleroom.services.orchestration_result_contract import expected_result_kind
from huddleroom.services.orchestration_work_report import CANONICAL_WORK_REPORT_SCHEMA, validate_work_report
from tests.test_orchestration_delegation_contracts import _agent, _delegation_request, _make_run
from tests.test_orchestration_plan_to_work_expansion import _make_accepted_plan, _raw_plan_item

CANONICAL = {"changed_files": "list[str]"}
SHAPES = [
    ("implementation", CANONICAL, "work_report", False),
    ("verification", {"verdict": "pass|fail", "reasons": "list[str]"}, "verification_verdict", False),
    ("final_summary", {"criteria": "list"}, "final_summary", False),
    ("decision", {"selected_option": "str"}, "decision_report", False),
    ("discovery", {"candidates": "list"}, "continuous_candidates", False),
]


def _schema_in_prompt(description: str) -> dict:
    tail = description.split("Read-only work reports its findings in this JSON.\n", 1)[1]
    obj, _ = json.JSONDecoder().raw_decode(tail)
    return obj


@pytest.mark.asyncio
@pytest.mark.parametrize("work_function,schema,kind,_gated", SHAPES)
async def test_delegation_persists_result_kind(db_session, test_project, work_function, schema, kind, _gated):
    agent = _agent("delegate", capabilities=[work_function])
    db_session.add(agent)
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, test_project.id)
    request = {**_delegation_request(agent.id), "work_function": work_function, "report_schema": schema}
    action = await service.execute_create_delegation_task_action(
        db_session, run_id=run.id, request=request,
        idempotency_key=f"run:{run.id}:kind:create_delegation_task:{work_function}",
    )
    task = await db_session.get(Task, action.id)
    contract = task.metadata_["orchestration_contract"]
    assert contract["expected_result"] == kind
    # the prompt names the kind's own schema: special kinds verbatim, work reports canonically
    shown = _schema_in_prompt(task.description)
    assert shown == (CANONICAL_WORK_REPORT_SCHEMA if kind == "work_report" else schema)
    expected_schema = CANONICAL_WORK_REPORT_SCHEMA if kind == "work_report" else schema
    fields = (
        'status "done" requires verdict accepted or rejected and evidence a list of strings with at least one '
        'item when accepted; use status "blocked" or "failed" with notes when you cannot verify; '
        "notes is optional and extra keys are tolerated (values show types). "
        if kind == "verification_verdict" else "no missing or extra keys (values show types). "
    )
    expected_tail = (
        "Required output:\nWhen done, return exactly one JSON object as your final message: no prose, "
        "no Markdown fences, " + fields
        + "Read-only work reports its findings in this JSON.\n" + json.dumps(expected_schema, indent=2)
    )
    if kind == "work_report":
        expected_tail += "\nUse only IDs given in this task; never invent IDs, and do not register Artifacts for this output."
    assert task.description.endswith(expected_tail)


@pytest.mark.asyncio
async def test_planning_task_persists_planning_envelope_and_keeps_planning_prompt(db_session, test_project):
    planner = _agent("planner", role="planner", capabilities=["planning"])
    db_session.add(planner)
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, test_project.id)
    action = await service.execute_request_plan_action(
        db_session, run_id=run.id,
        request={"action_type": "request_plan", "agent_id": str(planner.id), "work_function": "planning",
                 "scope": "Create a plan."},
        idempotency_key=f"run:{run.id}:kind:request_plan",
    )
    task = await db_session.get(Task, action.target_id)
    assert task.metadata_["orchestration_contract"]["expected_result"] == "planning_envelope"
    assert "Planning rules:" in task.description
    assert "Never invent task, agent, or Artifact UUIDs" in task.description


@pytest.mark.asyncio
async def test_gated_planning_role_item_gets_work_report_brief_and_first_report_is_consumed(db_session, test_project):
    planner = _agent("planner", role="planner", capabilities=["planning"])
    db_session.add(planner)
    await db_session.flush()
    service, _goal, run, _artifact = await _make_accepted_plan(
        db_session, test_project.id, planner,
        [_raw_plan_item(planner.id, item_id="checklist", work_function="planning")],
    )
    action = await service.execute_expand_plan_item_action(
        db_session, run_id=run.id,
        request={"action_type": "expand_plan_item", "plan_item_id": "checklist", "work_function": "planning"},
        idempotency_key=f"run:{run.id}:kind:expand_plan_item:plan_item:checklist",
    )
    task = await db_session.get(Task, action.target_id)
    contract = task.metadata_["orchestration_contract"]
    assert contract["expected_result"] == "work_report"
    assert contract["work_function"] == "planning"
    assert "Planning rules:" not in task.description
    assert _schema_in_prompt(task.description) == CANONICAL_WORK_REPORT_SCHEMA

    # A report built from exactly the brief's keys is what the validator accepts, and it is consumed first time.
    report = {key: ([] if shown.startswith("list") else {} if shown.startswith("dict") else None if "null" in shown else "x")
              for key, shown in _schema_in_prompt(task.description).items()}
    report["status"] = "completed"
    output = json.dumps(report)
    assert validate_work_report(output).report is not None
    orchestration = task.metadata_["orchestration"]
    task.metadata_ = {**task.metadata_, "orchestration": {**orchestration, "terminal_output": output}}
    task.status = "done"
    task.completed_at = _utcnow()
    await db_session.flush()
    await emit_event_once(db_session, test_project.id, "task.status_changed", {
        "task_id": str(task.id), "status": "done", "previous_status": "in_progress",
    }, dedup_key=f"gated-planning-report:{task.id}")
    event = await db_session.scalar(select(EventLog).where(EventLog.dedup_key == f"gated-planning-report:{task.id}"))
    await service._ingest_task_evidence(db_session, run, event)

    marker = await db_session.scalar(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == "report_consumed",
    ))
    assert marker.status == "completed"
    assert not any(b.get("kind") == "report_clarification_required" for b in run.active_blockers)
    assert await db_session.scalar(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == "report_clarification",
    )) is None


def test_legacy_contract_has_no_kind_key_and_still_derives():
    contract = OrchestrationDelegationContract(
        goal_id="00000000-0000-0000-0000-000000000001", run_id="00000000-0000-0000-0000-000000000002",
        action_id="00000000-0000-0000-0000-000000000003", agent_id="00000000-0000-0000-0000-000000000004",
        work_function="verification", scope="s", deliverable="d", report_schema={"verdict": "pass|fail"},
    )
    dumped = contract.model_dump(mode="json")
    assert "expected_result" not in dumped
    assert expected_result_kind(dumped, False) == "verification_verdict"


@pytest.mark.asyncio
async def test_gate_bound_planning_role_delegation_with_plan_kind_schema_stamps_work_report(db_session, test_project):
    agent = _agent("planner", role="planner", capabilities=["planning"])
    db_session.add(agent)
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, test_project.id)
    request = {**_delegation_request(agent.id), "work_function": "planning", "budget": {"max_tokens": 5000},
               "report_schema": {"plan_kind": "roadmap", "plan_items": "list"}}
    bound = await service.execute_create_delegation_task_action(
        db_session, run_id=run.id, request=request,
        idempotency_key=f"run:{run.id}:kind:create_delegation_task:roadmap_item:r1", gate_bound=True,
    )
    unbound = await service.execute_create_delegation_task_action(
        db_session, run_id=run.id, request={**request, "scope": "another scope"},
        idempotency_key=f"run:{run.id}:kind:create_delegation_task:roadmap_item:r2",
    )
    assert (await db_session.get(Task, bound.id)).metadata_["orchestration_contract"]["expected_result"] == "work_report"
    assert (await db_session.get(Task, unbound.id)).metadata_["orchestration_contract"]["expected_result"] == "planning_envelope"


@pytest.mark.asyncio
async def test_ungated_planning_role_task_without_plan_kind_is_consumed_as_work_report(db_session, test_project):
    agent = _agent("planner", role="planner", capabilities=["planning"])
    db_session.add(agent)
    await db_session.flush()
    service, _goal, run = await _make_run(db_session, test_project.id)
    action = await service.execute_create_delegation_task_action(
        db_session, run_id=run.id, request={**_delegation_request(agent.id), "work_function": "planning"},
        idempotency_key=f"run:{run.id}:kind:create_delegation_task:adhoc-planning",
    )
    task = await db_session.get(Task, action.id)
    assert task.metadata_["orchestration_contract"]["expected_result"] == "work_report"
    report = {key: ([] if shown.startswith("list") else {} if shown.startswith("dict") else None if "null" in shown else "x")
              for key, shown in CANONICAL_WORK_REPORT_SCHEMA.items()}
    report["status"] = "completed"
    task.metadata_ = {**task.metadata_, "orchestration": {**task.metadata_["orchestration"],
                                                          "terminal_output": json.dumps(report)}}
    task.status = "done"
    task.completed_at = _utcnow()
    await db_session.flush()
    await emit_event_once(db_session, test_project.id, "task.status_changed", {
        "task_id": str(task.id), "status": "done", "previous_status": "in_progress",
    }, dedup_key=f"ungated-planning-report:{task.id}")
    event = await db_session.scalar(select(EventLog).where(EventLog.dedup_key == f"ungated-planning-report:{task.id}"))
    await service._ingest_task_evidence(db_session, run, event)
    marker = await db_session.scalar(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == "report_consumed",
    ))
    assert marker is not None and marker.status == "completed"
    assert "plan_ingestion_error" not in (marker.request or {})
