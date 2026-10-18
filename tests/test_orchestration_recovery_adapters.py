"""SPR 180/184: existing-action registry adapters and corrective classification.

Interface assumptions (recovery + spec "Tool and verifier registry" / "Corrective-action ownership"):

- ``RecoveryToolRegistry().execute_call(db, proposal, call, wrapper, authority_decision) -> ToolReceiptV1``
  (the existing stub signature) is the registry entry point.  For the eight listed existing-action
  tools it builds a validated synthetic ``OrchestrationDecision`` (persisted in the run, accepted,
  ``applies_decision_id`` = the answered ``approve_fix`` decision) and dispatches it through
  ``OrchestrationDecisionDispatcher.dispatch(..., recovery_context=RecoveryApplyDispatchContext(...))``.
  ``team_configuration.apply`` goes straight to 's ``RecoveryConfigurationService.apply`` and never
  enters the dispatcher.
- ``OrchestrationService().classify_corrective_action(db, run, action_type, request, *, source)``
  returns ``CorrectiveDisposition``; ``request`` is the canonical decision request and ``source`` is a
  plain string naming the selecting path (tests use ``"model"`` and ``"recover_run"``; the disposition
  must come from durable state, not from that label).
"""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from uuid import UUID

import pytest
import pytest_asyncio
from fastapi import HTTPException
from sqlalchemy import func, select

from huddleroom.models.agent import Agent
from huddleroom.models.graph import Graph
from huddleroom.models.orchestration import (
    OrchestrationAction, OrchestrationDecision, OrchestrationGate, OrchestrationRun,
)
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.schemas.orchestration_recovery import (
    CorrectiveDisposition,
    ProposalDraftV1,
    RecoveryProposalV1,
    ToolReceiptV1,
)
from huddleroom.services.orchestration_blocker_recovery import _steering_seal
from huddleroom.services.orchestration_decision_dispatcher import OrchestrationDecisionDispatcher
from huddleroom.services.orchestration_recovery_configuration import RecoveryConfigurationService
from huddleroom.services.orchestration_recovery_registry import RecoveryToolRegistry
from huddleroom.services.orchestration_service import OrchestrationService
from tests.conftest import adopt_session
from tests.test_orchestration_recovery_configuration import (  # noqa: F401 -- fixtures re-exported
    approved_flow,
    load_flow,
    plan_author_or_none,
    seed_spr180,
    sessions,
)
from tests.test_orchestration_recovery_registry import (
    PLANNER_ID,
    REVIEWER_ID,
    _agent,
    _seed_goal_run,
    _trigger,
)

SOURCES = ("model", "recover_run")
ADAPTER_TOOLS = (
    "orchestration.request_plan", "orchestration.request_plan_revision", "orchestration.request_roadmap_replan",
    "orchestration.retry_task", "orchestration.reassign_task", "orchestration.request_verification",
    "orchestration.schedule_meeting", "orchestration.start_graph",
)
ACTION_TYPES = {tool: tool.split(".", 1)[1] for tool in ADAPTER_TOOLS}


@pytest_asyncio.fixture
async def db_session(owned_session_factory):
    """Autobegin session (no session.begin()) so the test task can own it."""
    async with owned_session_factory() as db:
        yield db
        await db.rollback()


def sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


# --------------------------------------------------------------------------- seeding


async def seed_rows(db, project, *, task_status="ready", gate_status="open"):
    adopt_session(db)  # the test task owns this autobegin session, as get_db does
    planner = Agent(**_agent(PLANNER_ID, "Planner", "planning").__dict__)
    organizer = Agent(**_agent(REVIEWER_ID, "Organizer", "review").__dict__)
    db.add_all([planner, organizer])
    goal, run = await _seed_goal_run(db, project, objective="Deliver the release.")
    gate = OrchestrationGate(run_id=run.id, success_criterion_key="plan", gate_type="plan", status=gate_status)
    graph = Graph(project_id=project.id, name="repair graph", version="1", definition={"nodes": []},
                  triggers=[], is_active=True)
    db.add_all([gate, graph])
    await db.flush()
    task = Task(project_id=project.id, title="Delivery work", assigned_to=planner.id, status=task_status,
                metadata_={"orchestration": {"run_id": str(run.id), "work_function": "planning",
                                             "gate_id": str(gate.id)}})
    db.add(task)
    await db.flush()
    return SimpleNamespace(goal=goal, run=run, planner=planner, organizer=organizer, gate=gate, graph=graph, task=task)


def tool_arguments(tool: str, rows) -> dict:
    return {
        "orchestration.request_plan": {"agent_id": str(rows.planner.id), "work_function": "planning", "scope": "goal"},
        "orchestration.request_plan_revision": {"plan_task_id": str(rows.task.id), "revision_request": "Repair plan."},
        "orchestration.request_roadmap_replan": {"agent_id": str(rows.planner.id), "scope": "goal", "reason": "Repair roadmap."},
        "orchestration.retry_task": {"task_id": str(rows.task.id)},
        "orchestration.reassign_task": {"task_id": str(rows.task.id), "agent_id": str(rows.organizer.id)},
        "orchestration.request_verification": {"gate_id": str(rows.gate.id), "work_function": "validation"},
        "orchestration.schedule_meeting": {"topic": "Repair", "participant_agent_ids": [str(rows.planner.id), str(rows.organizer.id)],
                                           "organizer_agent_id": str(rows.organizer.id), "task_id": str(rows.task.id),
                                           "gate_id": str(rows.gate.id)},
        "orchestration.start_graph": {"graph_id": str(rows.graph.id), "subject_type": "task", "subject_id": str(rows.task.id)},
    }[tool]


async def approved_call(db, rows, tool: str, arguments: dict | None = None):
    """Seal a real proposal, persist it like , answer approve_fix and reserve the apply wrapper."""
    label = ACTION_TYPES[tool].replace("_", " ")
    draft = ProposalDraftV1.model_validate({
        "kind": "proposal", "diagnosis": "The blocked work needs an existing orchestration action.",
        "evidence": [{"label": "The run is blocked.", "value": {"status": "blocked"}, "source_kind": "run",
                      "source_id": str(rows.run.id)}],
        "problem": "Work cannot continue.", "general_updates": [label],
        "expected_effect": "The blocked work can continue.",
        "calls": [{"tool": f"{tool}@1", "arguments": arguments or tool_arguments(tool, rows),
                   "rationale": "Repair the blocker."}],
    })
    proposal = await RecoveryToolRegistry().canonicalize(db, rows.goal, rows.run, _trigger(rows.goal.id, rows.run.id), draft)
    request = proposal.model_dump(mode="json")
    proposal_hash = sha(request)
    service = OrchestrationService()
    proposal_action = await service.reserve_action(
        db, rows.run.id, f"recovery:{proposal.proposal_id}:proposal", "recovery_proposal", request,
    )
    proposal_action.dispatch_contract = {
        "proposal_hash": proposal_hash, "resource_fingerprint": proposal.resource_fingerprint,
        "lineage_id": str(proposal.lineage_id), "attempt": proposal.attempt, "registry_version": "1",
        "steering": await _steering_seal(db, rows.goal, rows.run),
    }
    authority = OrchestrationAuthorityDecision(
        goal_id=rows.goal.id, run_id=rows.run.id, decision_key=f"recovery:{proposal.proposal_id}:{proposal_hash}",
        title="Approve recovery fix", authority="human", question="Apply the fix?",
        options=["approve_fix", "reject"], status="answered", selected_option="approve_fix",
        related_action_id=proposal_action.id,
    )
    db.add(authority)
    await db.flush()
    call = proposal.calls[0]
    wrapper = await service.reserve_action(
        db, rows.run.id, f"recovery:{proposal.proposal_id}:apply:{call.call_id}", "recovery_apply",
        {"proposal_action_id": str(proposal_action.id), "proposal_hash": proposal_hash, "call_id": call.call_id},
    )
    wrapper.dispatch_contract = {"authority_decision_id": str(authority.id), "proposal_hash": proposal_hash,
                                 "canonical_request_hash": sha(call.arguments)}
    await db.flush()
    return SimpleNamespace(proposal=proposal, call=call, wrapper=wrapper, authority=authority,
                           proposal_action=proposal_action, proposal_hash=proposal_hash)


class ExecutorRecorder:
    """Replaces the existing executor bodies with a real reserve/complete so dispatch persistence stays real."""

    def __init__(self, monkeypatch, rows, *, reject: HTTPException | None = None):
        self.calls: list[dict] = []
        self.rows = rows
        self.reject = reject
        for action_type, method in OrchestrationDecisionDispatcher._EXECUTORS.items():
            monkeypatch.setattr(OrchestrationService, method, self._executor(action_type))

    def _executor(self, action_type):
        recorder = self

        async def execute(service, db, run_id, request, idempotency_key, decision_id=None, **extra):
            recorder.calls.append({"action_type": action_type, "run_id": run_id, "request": request,
                                   "idempotency_key": idempotency_key, "decision_id": decision_id, "extra": extra})
            if recorder.reject is not None:
                db.add(Task(project_id=recorder.rows.goal.project_id, title="partial executor work"))
                await db.flush()
                raise recorder.reject
            action = await service.reserve_action(db, run_id, idempotency_key, action_type, request, decision_id)
            if action.status != "reserved":
                return action
            return await service._mark_action_completed(db, action, "task", recorder.rows.task.id)

        return execute


async def run_actions(db, run_id, *, action_type=None, suffix=None):
    query = select(OrchestrationAction).where(OrchestrationAction.run_id == run_id)
    if action_type:
        query = query.where(OrchestrationAction.action_type == action_type)
    rows = (await db.scalars(query.execution_options(populate_existing=True))).all()
    return [row for row in rows if suffix is None or row.idempotency_key.endswith(suffix)]


def spy_dispatch(monkeypatch):
    seen: list[dict] = []
    original = OrchestrationDecisionDispatcher.dispatch

    async def dispatch(self, db, run, decision, *, recovery_context=None):
        seen.append({"service": self._service, "decision": decision, "context": recovery_context})
        return await original(self, db, run, decision, recovery_context=recovery_context)

    monkeypatch.setattr(OrchestrationDecisionDispatcher, "dispatch", dispatch)
    return seen


# --------------------------------------------------------------------------- adapter parity


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ADAPTER_TOOLS)
async def test_existing_action_adapter_dispatches_canonical_request_through_dispatcher(
    db_session, test_project, monkeypatch, tool,
):
    rows = await seed_rows(db_session, test_project, task_status="failed")
    if tool == "orchestration.request_plan":
        rows.run.plan_state = {"test_nested": {"items": []}}
        await db_session.flush()
    approved = await approved_call(db_session, rows, tool)
    recorder = ExecutorRecorder(monkeypatch, rows)
    seen = spy_dispatch(monkeypatch)

    receipt = await RecoveryToolRegistry().execute_call(
        db_session, approved.proposal, approved.call, approved.wrapper, approved.authority,
    )

    action_type = ACTION_TYPES[tool]
    wrapper_key = f"recovery:{approved.proposal.proposal_id}:apply:{approved.call.call_id}"
    # Exactly one trusted dispatch, with the server-only apply context bound to this approval.
    assert len(seen) == 1
    context = seen[0]["context"]
    assert context is not None and context.purpose == "apply"
    assert (context.proposal_id, context.call_id, context.authority_decision_id) == (
        approved.proposal.proposal_id, approved.call.call_id, approved.authority.id)
    decision = seen[0]["decision"]
    assert decision.validator_status == "accepted"
    assert decision.parsed_decision["action_type"] == action_type
    assert str(decision.parsed_decision["applies_decision_id"]) == str(approved.authority.id)
    stored = await db_session.get(OrchestrationDecision, decision.id)
    assert stored is not None and stored.run_id == rows.run.id
    if tool == "orchestration.request_plan":
        assert decision.input_snapshot == {"run": {
            "id": str(rows.run.id), "plan_state": {"test_nested": {"items": []}},
        }}
        rows.run.plan_state["test_nested"]["items"].append("later state mutation")
        assert decision.input_snapshot["run"]["plan_state"]["test_nested"]["items"] == []
    # Same canonical request and executor as an ordinary model decision would reach.
    ordinary_request = OrchestrationService().canonical_decision_request(
        action_type, {"action_type": action_type, **tool_arguments(tool, rows)}, run_id=rows.run.id,
    )
    assert approved.call.arguments == ordinary_request
    assert [call["action_type"] for call in recorder.calls] == [action_type]
    assert recorder.calls[0]["request"] == ordinary_request
    assert recorder.calls[0]["idempotency_key"] == f"{wrapper_key}:effect"
    assert recorder.calls[0]["decision_id"] == decision.id
    # One linked ordinary domain-effect row, distinct from the lifecycle wrapper.
    effects = await run_actions(db_session, rows.run.id, action_type=action_type)
    assert len(effects) == 1
    effect = effects[0]
    assert effect.idempotency_key == f"{wrapper_key}:effect" and effect.id != approved.wrapper.id
    contract = approved.wrapper.dispatch_contract
    assert contract["effect_action_id"] == str(effect.id)
    assert contract["effect_idempotency_key"] == effect.idempotency_key
    assert contract["authority_decision_id"] == str(approved.authority.id)
    assert contract["canonical_request_hash"] == sha(approved.call.arguments)
    # Receipt binds wrapper and effect identities.
    assert isinstance(receipt, ToolReceiptV1)
    assert receipt.status == "applied"
    assert receipt.proposal_id == approved.proposal.proposal_id and receipt.call_id == approved.call.call_id
    assert receipt.tool == f"{tool}@1"
    assert receipt.wrapper_action_id == approved.wrapper.id
    assert receipt.action_id == effect.id
    assert receipt.idempotency_key == effect.idempotency_key
    assert receipt.canonical_request_hash == sha(approved.call.arguments)
    identity = receipt.execution_identity
    assert (identity.proposal_id, identity.call_id, identity.authority_decision_id, identity.purpose) == (
        approved.proposal.proposal_id, approved.call.call_id, approved.authority.id, "apply")
    assert {"kind": effect.target_type, "id": str(effect.target_id)} in receipt.target_refs


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ("orchestration.retry_task", "orchestration.request_plan"))
async def test_adapter_replay_reuses_the_one_linked_effect(db_session, test_project, monkeypatch, tool):
    rows = await seed_rows(db_session, test_project, task_status="failed")
    approved = await approved_call(db_session, rows, tool)
    ExecutorRecorder(monkeypatch, rows)
    registry = RecoveryToolRegistry()

    first = await registry.execute_call(db_session, approved.proposal, approved.call, approved.wrapper, approved.authority)
    second = await registry.execute_call(db_session, approved.proposal, approved.call, approved.wrapper, approved.authority)

    assert second.action_id == first.action_id and second.wrapper_action_id == first.wrapper_action_id
    assert len(await run_actions(db_session, rows.run.id, action_type=ACTION_TYPES[tool])) == 1
    assert len(await run_actions(db_session, rows.run.id, action_type="recovery_apply")) == 1


@pytest.mark.asyncio
async def test_adapter_rejection_matches_ordinary_dispatch_rejection(db_session, test_project, monkeypatch):
    """A genuine executor 4xx rolls back partial work and leaves one failed effect row, exactly like ordinary dispatch."""
    rows = await seed_rows(db_session, test_project)
    approved = await approved_call(db_session, rows, "orchestration.request_plan")
    ExecutorRecorder(monkeypatch, rows, reject=HTTPException(status_code=409, detail="plan_conflict"))
    tasks_before = await db_session.scalar(select(func.count(Task.id)))

    receipt = await RecoveryToolRegistry().execute_call(
        db_session, approved.proposal, approved.call, approved.wrapper, approved.authority,
    )

    effect_key = f"recovery:{approved.proposal.proposal_id}:apply:{approved.call.call_id}:effect"
    effects = await run_actions(db_session, rows.run.id, action_type="request_plan")
    assert [(row.idempotency_key, row.status, row.error) for row in effects] == [(effect_key, "failed", "plan_conflict")]
    assert await db_session.scalar(select(func.count(Task.id))) == tasks_before
    assert receipt.status == "failed"
    assert receipt.action_id == effects[0].id and receipt.wrapper_action_id == approved.wrapper.id

    # Ordinary dispatch of the same request: same failed-row semantics and error.
    ordinary = OrchestrationDecision(
        run_id=rows.run.id, decision_type="execution", validator_status="accepted",
        input_snapshot={"run": {"id": str(rows.run.id), "plan_state": rows.run.plan_state}},
        parsed_decision={"action_type": "request_plan", **tool_arguments("orchestration.request_plan", rows)},
    )
    db_session.add(ordinary)
    await db_session.flush()
    ordinary_action = await OrchestrationDecisionDispatcher(OrchestrationService()).dispatch(db_session, rows.run, ordinary)
    assert (ordinary_action.status, ordinary_action.error) == ("failed", "plan_conflict")
    assert ordinary_action.request == effects[0].request
    assert await db_session.scalar(select(func.count(Task.id))) == tasks_before


@pytest.mark.asyncio
async def test_adapter_never_dispatches_arguments_that_differ_from_the_sealed_call(db_session, test_project, monkeypatch):
    rows = await seed_rows(db_session, test_project, task_status="failed")
    approved = await approved_call(db_session, rows, "orchestration.reassign_task")
    recorder = ExecutorRecorder(monkeypatch, rows)
    tampered = approved.call.model_copy(update={"arguments": {**approved.call.arguments, "agent_id": str(rows.planner.id)}})

    try:
        outcome = await RecoveryToolRegistry().execute_call(
            db_session, approved.proposal, tampered, approved.wrapper, approved.authority,
        )
    except (ValueError, HTTPException) as exc:
        outcome = exc

    assert not (isinstance(outcome, ToolReceiptV1) and outcome.status == "applied")
    assert recorder.calls == []
    assert await run_actions(db_session, rows.run.id, suffix=":effect") == []
    assert (await db_session.get(Task, rows.task.id)).assigned_to == rows.planner.id


@pytest.mark.asyncio
async def test_adapter_uses_real_verification_executor_and_target(
    db_session, test_project, safe_effectiveness_review_continue,
):
    """No stubs: the effect is the real verification executor's action and its real verifier task target."""
    from tests.test_orchestration_decision_dispatcher import _verifiable_gate

    adopt_session(db_session)
    service, goal, run, gate = await _verifiable_gate(db_session, test_project)
    planner = Agent(**_agent(PLANNER_ID, "Planner", "planning").__dict__)
    db_session.add(planner)
    await db_session.flush()
    rows = SimpleNamespace(goal=goal, run=run, gate=gate, planner=planner)
    approved = await approved_call(db_session, rows, "orchestration.request_verification",
                                   {"gate_id": str(gate.id), "work_function": "validation"})

    receipt = await RecoveryToolRegistry().execute_call(
        db_session, approved.proposal, approved.call, approved.wrapper, approved.authority,
    )

    effect = await db_session.get(OrchestrationAction, receipt.action_id)
    assert receipt.status == "applied"
    assert effect.action_type == "request_verification" and effect.status == "completed"
    assert effect.idempotency_key == f"{approved.wrapper.idempotency_key}:effect"
    verifier_task = await db_session.get(Task, effect.target_id)
    assert verifier_task is not None and verifier_task.metadata_["orchestration"]["run_id"] == str(run.id)
    assert {"kind": "task", "id": str(verifier_task.id)} in receipt.target_refs


@pytest.mark.asyncio
async def test_configuration_aggregate_never_enters_dispatcher_and_owns_its_effect(sessions, tmp_path, monkeypatch):
    seeded = await seed_spr180(sessions, tmp_path)
    flow = await approved_flow(sessions, seeded)

    async def no_dispatch(*_args, **_kwargs):
        raise AssertionError("team_configuration.apply must not enter the ordinary dispatcher")

    monkeypatch.setattr(OrchestrationDecisionDispatcher, "dispatch", no_dispatch)
    applied = []
    original_apply = RecoveryConfigurationService.apply

    async def apply(self, *args, **kwargs):
        applied.append(args)
        return await original_apply(self, *args, **kwargs)

    monkeypatch.setattr(RecoveryConfigurationService, "apply", apply)

    async with sessions() as s:
        proposal, call, wrapper, authority = await load_flow(s, flow)
        receipt = await RecoveryToolRegistry().execute_call(s, proposal, call, wrapper, authority)

    assert len(applied) == 1
    assert receipt.status == "applied"
    assert receipt.action_id == receipt.wrapper_action_id == flow.wrapper_id
    assert receipt.idempotency_key == f"recovery:{proposal.proposal_id}:apply:{call.call_id}"
    async with sessions() as s:
        effect_rows = [row for row in (await s.scalars(select(OrchestrationAction).where(
            OrchestrationAction.run_id == seeded.run_id))).all() if row.idempotency_key.endswith(":effect")]
        assert effect_rows == []
        assert (await s.get(OrchestrationAction, flow.wrapper_id)).status == "completed"
        assert await plan_author_or_none(s, seeded.goal_id, seeded.run_id) is not None


# --------------------------------------------------------------------------- corrective classification


async def classify(db, run, action_type, arguments, source):
    service = OrchestrationService()
    request = service.canonical_decision_request(action_type, {"action_type": action_type, **arguments}, run_id=run.id)
    return await service.classify_corrective_action(db, run, action_type, request, source=source)


async def durable_state(db, rows):
    task = await db.get(Task, rows.task.id, populate_existing=True)
    actions = await db.scalar(select(func.count(OrchestrationAction.id)))
    tasks = await db.scalar(select(func.count(Task.id)))
    session_count = await db.scalar(select(func.count(Session.id)))
    return task.status, task.assigned_to, task.metadata_, actions, tasks, session_count


def delegation_arguments(rows, **extra):
    return {"agent_id": str(rows.planner.id), "work_function": "implementation", "scope": "Implement the change.",
            "deliverable": "A tested patch.", **extra}


@pytest.mark.asyncio
@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("action_type", (
    "request_plan", "request_verification", "schedule_meeting", "start_graph", "create_delegation_task",
))
async def test_healthy_initial_work_remains_ordinary(db_session, test_project, action_type, source):
    rows = await seed_rows(db_session, test_project)
    arguments = delegation_arguments(rows) if action_type == "create_delegation_task" else tool_arguments(
        f"orchestration.{action_type}", rows)
    before = await durable_state(db_session, rows)

    disposition = await classify(db_session, rows.run, action_type, arguments, source)

    assert disposition == CorrectiveDisposition.ORDINARY
    assert await durable_state(db_session, rows) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("source", SOURCES)
async def test_durable_failure_retry_requires_recovery_approval(db_session, test_project, source):
    rows = await seed_rows(db_session, test_project, task_status="failed")
    task = rows.task
    original_assignee = task.assigned_to
    before = await durable_state(db_session, rows)

    disposition = await classify(db_session, rows.run, "retry_task", {"task_id": str(task.id)}, source)

    assert disposition == CorrectiveDisposition.REQUIRES_RECOVERY
    assert task.assigned_to == original_assignee
    assert await durable_state(db_session, rows) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("source", SOURCES)
async def test_durable_failure_reassignment_requires_recovery_and_keeps_assignee(db_session, test_project, source):
    rows = await seed_rows(db_session, test_project, task_status="failed")
    original_assignee = rows.task.assigned_to
    before = await durable_state(db_session, rows)

    disposition = await classify(db_session, rows.run, "reassign_task",
                                 {"task_id": str(rows.task.id), "agent_id": str(rows.organizer.id)}, source)

    assert disposition == CorrectiveDisposition.REQUIRES_RECOVERY
    assert (await db_session.get(Task, rows.task.id)).assigned_to == original_assignee
    assert await durable_state(db_session, rows) == before


@pytest.mark.asyncio
@pytest.mark.parametrize(("action_type", "arguments"), (
    ("retry_task", lambda rows: {"task_id": str(rows.task.id), "max_tokens": 5000}),
    ("reassign_task", lambda rows: {"task_id": str(rows.task.id), "agent_id": str(rows.organizer.id)}),
))
async def test_retry_and_reassign_without_durable_failure_are_still_corrective(db_session, test_project, action_type, arguments):
    """retry/reassign are always corrective except for reconciling the identical approved attempt."""
    rows = await seed_rows(db_session, test_project, task_status="blocked")

    assert await classify(db_session, rows.run, action_type, arguments(rows), "model") == (
        CorrectiveDisposition.REQUIRES_RECOVERY)


@pytest.mark.asyncio
@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("case", (
    "verification_after_failed_gate", "meeting_on_failed_task", "graph_on_failed_task",
    "plan_revision_after_failed_plan_task", "roadmap_replan_after_failed_task", "replacement_delegation_for_failed_task",
))
async def test_corrective_mutation_after_durable_failure_requires_recovery(db_session, test_project, case, source):
    failed_gate = case == "verification_after_failed_gate"
    rows = await seed_rows(db_session, test_project, task_status="ready" if failed_gate else "failed",
                           gate_status="failed" if failed_gate else "open")
    action_type, arguments = {
        "verification_after_failed_gate": ("request_verification", tool_arguments("orchestration.request_verification", rows)),
        "meeting_on_failed_task": ("schedule_meeting", tool_arguments("orchestration.schedule_meeting", rows)),
        "graph_on_failed_task": ("start_graph", tool_arguments("orchestration.start_graph", rows)),
        "plan_revision_after_failed_plan_task": ("request_plan_revision", tool_arguments("orchestration.request_plan_revision", rows)),
        "roadmap_replan_after_failed_task": ("request_roadmap_replan", tool_arguments("orchestration.request_roadmap_replan", rows)),
        "replacement_delegation_for_failed_task": ("create_delegation_task", delegation_arguments(rows, parent_task_id=str(rows.task.id))),
    }[case]
    before = await durable_state(db_session, rows)

    disposition = await classify(db_session, rows.run, action_type, arguments, source)

    assert disposition == CorrectiveDisposition.REQUIRES_RECOVERY
    assert await durable_state(db_session, rows) == before


async def pending_approved_retry_effect(db, rows):
    """An approved recovery retry whose effect was reserved, then the worker crashed before running it."""
    approved = await approved_call(db, rows, "orchestration.retry_task")
    effect_key = f"{approved.wrapper.idempotency_key}:effect"
    effect = await OrchestrationService().reserve_action(db, rows.run.id, effect_key, "retry_task", approved.call.arguments)
    approved.wrapper.dispatch_contract = {
        **approved.wrapper.dispatch_contract, "effect_action_id": str(effect.id), "effect_idempotency_key": effect_key,
    }
    await db.flush()
    return approved, effect


@pytest.mark.asyncio
@pytest.mark.parametrize("source", SOURCES)
async def test_identical_pending_effect_reconciliation_is_not_a_new_fix(db_session, test_project, source):
    rows = await seed_rows(db_session, test_project, task_status="failed")
    _approved, effect = await pending_approved_retry_effect(db_session, rows)
    assert effect.status == "reserved"
    before = await durable_state(db_session, rows)

    disposition = await classify(db_session, rows.run, "retry_task", {"task_id": str(rows.task.id)}, source)

    assert disposition == CorrectiveDisposition.SAME_ATTEMPT_RECONCILIATION
    assert await durable_state(db_session, rows) == before


@pytest.mark.asyncio
async def test_pending_effect_with_a_different_request_is_a_new_corrective_attempt(db_session, test_project):
    """Identity is the canonical request, not the action type: a new budget is a new fix."""
    rows = await seed_rows(db_session, test_project, task_status="failed")
    await pending_approved_retry_effect(db_session, rows)

    disposition = await classify(db_session, rows.run, "retry_task",
                                 {"task_id": str(rows.task.id), "max_tokens": 9000}, "model")

    assert disposition == CorrectiveDisposition.REQUIRES_RECOVERY


@pytest.mark.asyncio
async def test_completed_approved_attempt_followed_by_new_failure_requires_recovery(db_session, test_project):
    rows = await seed_rows(db_session, test_project, task_status="failed")
    _approved, effect = await pending_approved_retry_effect(db_session, rows)
    await OrchestrationService()._mark_action_completed(db_session, effect, "task", rows.task.id)

    disposition = await classify(db_session, rows.run, "retry_task", {"task_id": str(rows.task.id)}, "recover_run")

    assert disposition == CorrectiveDisposition.REQUIRES_RECOVERY


@pytest.mark.asyncio
async def test_approved_fix_for_one_failure_does_not_make_an_unrelated_same_type_failure_ordinary(
    db_session, test_project,
):
    """F1: a completed approved effect recovers only the failure its lineage names (or its exact identity); an
    unrelated failed action of the same TYPE stays corrective and still needs its own approved fix."""
    from datetime import datetime, timedelta, timezone
    from uuid import uuid4

    rows = await seed_rows(db_session, test_project)
    service = OrchestrationService()
    run = rows.run

    def canonical(gate_id):
        return service.canonical_decision_request(
            "request_verification", {"action_type": "request_verification", "gate_id": gate_id,
                                     "work_function": "validation"}, run_id=run.id)

    request_a, request_b = canonical(str(rows.gate.id)), canonical(str(uuid4()))
    start = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=10)
    failed = {}
    for offset, (name, request) in enumerate((("a", request_a), ("b", request_b))):
        failed[name] = OrchestrationAction(
            id=uuid4(), run_id=run.id,
            idempotency_key=OrchestrationDecisionDispatcher.action_key(run.id, "request_verification", request),
            action_type="request_verification", request=request, status="failed", error="verifier_unavailable",
            created_at=start + timedelta(minutes=offset))
    pid = uuid4()
    db_session.add_all([*failed.values(), OrchestrationAction(
        run_id=run.id, idempotency_key=f"recovery:{pid}:proposal", action_type="recovery_proposal", status="completed",
        request={"proposal_id": str(pid), "retry": {"target": {"kind": "action", "source_action_id": str(failed["a"].id)}},
                 "source": {"kind": "failed_action", "id": str(failed["a"].id)}},
        created_at=start + timedelta(minutes=2),
    )])
    await db_session.flush()
    db_session.add(OrchestrationAction(
        run_id=run.id, idempotency_key=f"recovery:{pid}:retry:1:effect", action_type="request_verification",
        request={**request_a, "work_function": "validation", "note": "sealed retry"}, status="completed",
        created_at=start + timedelta(minutes=3),
    ))
    await db_session.flush()

    assert await service.classify_corrective_action(
        db_session, run, "request_verification", request_a, source="model") == CorrectiveDisposition.ORDINARY
    assert await service.classify_corrective_action(
        db_session, run, "request_verification", request_b, source="model") == CorrectiveDisposition.REQUIRES_RECOVERY


@pytest.mark.asyncio
async def test_transport_retry_of_identical_pending_ordinary_effect_is_reconciliation(db_session, test_project):
    rows = await seed_rows(db_session, test_project)
    service = OrchestrationService()
    request = service.canonical_decision_request(
        "request_verification",
        {"action_type": "request_verification", **tool_arguments("orchestration.request_verification", rows)},
        run_id=rows.run.id,
    )
    pending = await service.reserve_action(
        db_session, rows.run.id, OrchestrationDecisionDispatcher.action_key(rows.run.id, "request_verification", request),
        "request_verification", request,
    )
    assert pending.status == "reserved"

    disposition = await service.classify_corrective_action(
        db_session, rows.run, "request_verification", request, source="model",
    )

    assert disposition == CorrectiveDisposition.SAME_ATTEMPT_RECONCILIATION


# --------------------------------------------------------------------------- follow-up


@pytest.mark.asyncio
async def test_pending_unapproved_ordinary_retry_after_failure_still_requires_recovery(db_session, test_project):
    """I1: only the identical APPROVED pending attempt may resume a corrective action."""
    rows = await seed_rows(db_session, test_project, task_status="failed")
    service = OrchestrationService()
    request = service.canonical_decision_request("retry_task", {"action_type": "retry_task", "task_id": str(rows.task.id)},
                                                 run_id=rows.run.id)
    await service.reserve_action(db_session, rows.run.id,
                                 OrchestrationDecisionDispatcher.action_key(rows.run.id, "retry_task", request),
                                 "retry_task", request)

    assert await service.classify_corrective_action(db_session, rows.run, "retry_task", request, source="model") == (
        CorrectiveDisposition.REQUIRES_RECOVERY)


@pytest.mark.asyncio
async def test_meeting_on_blocked_task_requires_recovery(db_session, test_project):
    """I2: an active blocker is a corrective source."""
    rows = await seed_rows(db_session, test_project, task_status="blocked")

    assert await classify(db_session, rows.run, "schedule_meeting", tool_arguments("orchestration.schedule_meeting", rows),
                          "model") == CorrectiveDisposition.REQUIRES_RECOVERY


@pytest.mark.asyncio
async def test_new_attempt_after_failed_ordinary_action_requires_recovery(db_session, test_project):
    """I2: a durably failed ordinary action with the same request makes the next attempt corrective."""
    rows = await seed_rows(db_session, test_project)
    service = OrchestrationService()
    arguments = tool_arguments("orchestration.request_verification", rows)
    request = service.canonical_decision_request("request_verification", {"action_type": "request_verification", **arguments},
                                                 run_id=rows.run.id)
    failed = await service.reserve_action(
        db_session, rows.run.id, OrchestrationDecisionDispatcher.action_key(rows.run.id, "request_verification", request),
        "request_verification", request)
    await service._fail_reserved_action_in_transaction(db_session, failed, "verifier_unavailable")

    assert await classify(db_session, rows.run, "request_verification", arguments, "model") == (
        CorrectiveDisposition.REQUIRES_RECOVERY)


@pytest.mark.asyncio
async def test_graph_on_recovery_produced_target_is_ordinary_until_it_fails(db_session, test_project):
    """Healthy work on a recovery-produced target is
    ordinary; once that target durably fails, the next action on it is corrective."""
    rows = await seed_rows(db_session, test_project)
    lineage = await OrchestrationService().reserve_action(
        db_session, rows.run.id, "recovery:00000000-0000-0000-0000-000000000001:apply:c1:effect", "retry_task",
        {"task_id": str(rows.task.id)})
    await OrchestrationService()._mark_action_completed(db_session, lineage, "task", rows.task.id)

    assert await classify(db_session, rows.run, "start_graph", tool_arguments("orchestration.start_graph", rows),
                          "recover_run") == CorrectiveDisposition.ORDINARY
    rows.task.status = "failed"
    await db_session.flush()
    assert await classify(db_session, rows.run, "start_graph", tool_arguments("orchestration.start_graph", rows),
                          "recover_run") == CorrectiveDisposition.REQUIRES_RECOVERY


@pytest.mark.asyncio
async def test_adapter_replay_of_linked_pending_effect_is_not_redispatched(db_session, test_project, monkeypatch):
    """I3: a linked effect that is still reserved yields effect_unknown; the executor never runs again."""
    rows = await seed_rows(db_session, test_project, task_status="failed")
    approved, effect = await pending_approved_retry_effect(db_session, rows)
    recorder = ExecutorRecorder(monkeypatch, rows)
    seen = spy_dispatch(monkeypatch)

    receipt = await RecoveryToolRegistry().execute_call(
        db_session, approved.proposal, approved.call, approved.wrapper, approved.authority,
    )

    assert receipt.status == "effect_unknown" and receipt.action_id == effect.id
    assert recorder.calls == [] and seen == []
    assert (await db_session.get(OrchestrationAction, effect.id)).status == "reserved"
    assert "tool_receipt_v1" not in approved.wrapper.dispatch_contract


@pytest.mark.asyncio
async def test_steering_change_after_approval_blocks_apply(db_session, test_project, monkeypatch):
    """I4: the resource fingerprint does not cover steering, so apply fences the versions sealed at proposal time."""
    from huddleroom.services.orchestration_blocker_recovery import _steering_seal
    from huddleroom.services.orchestration_steering import SteeringVersionsChanged

    rows = await seed_rows(db_session, test_project, task_status="failed")
    approved = await approved_call(db_session, rows, "orchestration.retry_task")
    seal = await _steering_seal(db_session, rows.goal, rows.run)
    assert seal["versions"]
    approved.proposal_action.dispatch_contract = {**approved.proposal_action.dispatch_contract, "steering": seal}
    rows.run.plan_state = {**(rows.run.plan_state or {}), "accepted_plan_fingerprint": "changed-after-approval"}
    await db_session.flush()
    recorder = ExecutorRecorder(monkeypatch, rows)

    with pytest.raises(SteeringVersionsChanged):
        await RecoveryToolRegistry().execute_call(
            db_session, approved.proposal, approved.call, approved.wrapper, approved.authority,
        )

    assert recorder.calls == []
    assert await run_actions(db_session, rows.run.id, suffix=":effect") == []


@pytest.mark.asyncio
async def test_unchanged_steering_seal_lets_apply_proceed(db_session, test_project, monkeypatch):
    from huddleroom.services.orchestration_blocker_recovery import _steering_seal

    rows = await seed_rows(db_session, test_project, task_status="failed")
    approved = await approved_call(db_session, rows, "orchestration.retry_task")
    approved.proposal_action.dispatch_contract = {
        **approved.proposal_action.dispatch_contract, "steering": await _steering_seal(db_session, rows.goal, rows.run)}
    await db_session.flush()
    ExecutorRecorder(monkeypatch, rows)

    receipt = await RecoveryToolRegistry().execute_call(
        db_session, approved.proposal, approved.call, approved.wrapper, approved.authority,
    )

    assert receipt.status == "applied"


@pytest.mark.asyncio
async def test_adapter_budget_cap_rejection_matches_ordinary_dispatch(db_session, test_project):
    """M6: a real budget-cap 4xx from the real start_graph executor is a failed receipt, same as ordinary dispatch."""
    rows = await seed_rows(db_session, test_project)
    rows.run.budget_state = {"caps": {"max_tokens": "1000"}}
    await db_session.flush()
    approved = await approved_call(db_session, rows, "orchestration.start_graph")

    receipt = await RecoveryToolRegistry().execute_call(
        db_session, approved.proposal, approved.call, approved.wrapper, approved.authority,
    )

    effect = await db_session.get(OrchestrationAction, receipt.action_id)
    detail = "Budgeted control provider requires measured session telemetry"
    assert receipt.status == "failed" and receipt.error == detail
    assert (effect.status, effect.error) == ("failed", detail)
    ordinary = OrchestrationDecision(
        run_id=rows.run.id, decision_type="execution", validator_status="accepted", input_snapshot={},
        parsed_decision={"action_type": "start_graph", **tool_arguments("orchestration.start_graph", rows)},
    )
    db_session.add(ordinary)
    await db_session.flush()
    ordinary_action = await OrchestrationDecisionDispatcher(OrchestrationService()).dispatch(db_session, rows.run, ordinary)
    assert (ordinary_action.status, ordinary_action.error) == ("failed", detail)
    assert ordinary_action.request == effect.request


# --------------------------------------------------------------------------- follow-up: steering seal fails closed


async def _apply_with_seal(db, rows, monkeypatch, seal, *, drop=False):
    approved = await approved_call(db, rows, "orchestration.retry_task")
    contract = dict(approved.proposal_action.dispatch_contract)
    if drop:
        contract.pop("steering")
    else:
        contract["steering"] = seal
    approved.proposal_action.dispatch_contract = contract
    await db.flush()
    recorder = ExecutorRecorder(monkeypatch, rows)
    return approved, recorder


async def _assert_blocked(db, rows, approved, recorder):
    from huddleroom.services.orchestration_steering import SteeringVersionsChanged

    with pytest.raises(SteeringVersionsChanged):
        await RecoveryToolRegistry().execute_call(db, approved.proposal, approved.call, approved.wrapper, approved.authority)
    assert recorder.calls == []
    assert await run_actions(db, rows.run.id, suffix=":effect") == []


async def _active_direction(db, rows, user, *, status="applied"):
    from uuid import uuid4

    from huddleroom.models.orchestration_steering import OrchestrationSteeringRequest

    request = OrchestrationSteeringRequest(
        id=uuid4(), goal_id=rows.goal.id, actor_id=user.id, client_request_id=uuid4(), sequence=1,
        submitted_run_id=rows.run.id, directive="Prefer the smaller fix.", target_type="goal", target_id=str(rows.goal.id),
        scope="goal", lifetime="future_runs", impact_summary="Smaller fixes.", status=status,
        reason_code="submitted", contract_version="test", plan_version="test",
    )
    db.add(request)
    await db.flush()
    return request


@pytest.mark.asyncio
async def test_apply_without_steering_seal_fails_closed(db_session, test_project, monkeypatch):
    rows = await seed_rows(db_session, test_project, task_status="failed")
    approved, recorder = await _apply_with_seal(db_session, rows, monkeypatch, None, drop=True)
    await _assert_blocked(db_session, rows, approved, recorder)


@pytest.mark.asyncio
@pytest.mark.parametrize("seal", (
    {"versions": {"inbox_version": 0}},
    {"versions": None, "active_directions": []},
    "not-a-mapping",
    {"versions": {"inbox_version": 0, "direction_version": 0, "contract_version": "c", "plan_version": "p"},
     "active_directions": [{"no_request_id": True}]},
))
async def test_apply_with_invalid_steering_seal_fails_closed(db_session, test_project, monkeypatch, seal):
    rows = await seed_rows(db_session, test_project, task_status="failed")
    approved, recorder = await _apply_with_seal(db_session, rows, monkeypatch, seal)
    await _assert_blocked(db_session, rows, approved, recorder)


@pytest.mark.asyncio
async def test_empty_seal_fails_closed_once_steering_context_exists(db_session, test_project, monkeypatch):
    """{} is the disabled-steering seal; with steering enabled now, the current context is non-empty."""
    rows = await seed_rows(db_session, test_project, task_status="failed")
    approved, recorder = await _apply_with_seal(db_session, rows, monkeypatch, {})
    await _assert_blocked(db_session, rows, approved, recorder)


@pytest.mark.asyncio
async def test_empty_seal_passes_while_steering_is_disabled(db_session, test_project, monkeypatch):
    from huddleroom.config import settings

    monkeypatch.setattr(settings, "orchestration_conversation_steering_enabled", False)
    rows = await seed_rows(db_session, test_project, task_status="failed")
    approved, _recorder = await _apply_with_seal(db_session, rows, monkeypatch, {})
    assert approved.proposal_action.dispatch_contract["steering"] == {}

    receipt = await RecoveryToolRegistry().execute_call(
        db_session, approved.proposal, approved.call, approved.wrapper, approved.authority,
    )
    assert receipt.status == "applied"


@pytest.mark.asyncio
async def test_direction_added_after_approval_blocks_apply(db_session, test_project, test_user, monkeypatch):
    rows = await seed_rows(db_session, test_project, task_status="failed")
    approved, recorder = await _apply_with_seal(
        db_session, rows, monkeypatch, await _steering_seal(db_session, rows.goal, rows.run))
    await _active_direction(db_session, rows, test_user)
    await _assert_blocked(db_session, rows, approved, recorder)


@pytest.mark.asyncio
async def test_direction_removed_after_approval_blocks_apply(db_session, test_project, test_user, monkeypatch):
    rows = await seed_rows(db_session, test_project, task_status="failed")
    direction = await _active_direction(db_session, rows, test_user)
    seal = await _steering_seal(db_session, rows.goal, rows.run)
    assert [item["request_id"] for item in seal["active_directions"]] == [str(direction.id)]
    approved, recorder = await _apply_with_seal(db_session, rows, monkeypatch, seal)
    direction.status = "withdrawn"
    await db_session.flush()
    await _assert_blocked(db_session, rows, approved, recorder)


@pytest.mark.asyncio
async def test_diagnose_writes_the_steering_seal_on_the_proposal(db_session, conversation_goal_run, test_agent):
    from tests.test_orchestration_recovery_persistence import proposal_payload, service_for, trigger_for

    adopt_session(db_session)
    goal, run = conversation_goal_run
    run.status = "running"
    svc = service_for(proposal_payload(test_agent))
    trigger = await svc.create_trigger(db_session, trigger_for(goal, run))

    proposal = await svc.diagnose(db_session, trigger.id, tick_owns_transaction=True)

    seal = proposal.dispatch_contract["steering"]
    assert seal == await _steering_seal(db_session, goal, run)
    assert set(seal["versions"]) == {"inbox_version", "direction_version", "contract_version", "plan_version"}
    assert seal["active_directions"] == []


@pytest.mark.asyncio
async def test_waiver_is_only_autonomous_corrective_exception(db_session, test_project):
    rows = await seed_rows(db_session, test_project, task_status="failed")
    service = OrchestrationService()
    assert await service.classify_corrective_action(db_session, rows.run, "resolve_meeting_action_item", {"item_id": str(rows.task.id), "reason": "Irrelevant"}, source="model") == CorrectiveDisposition.ORDINARY
    assert await service.classify_corrective_action(db_session, rows.run, "retry_task", {"task_id": str(rows.task.id)}, source="model") == CorrectiveDisposition.REQUIRES_RECOVERY
