"""SPR 180/184: executable delegated remediation and named-gate baseline admission.

Interface assumptions (recovery + spec "Delegated remediation"):

- ``huddleroom.services.orchestration_agent_execution_evidence`` exports ``AgentExecutionEvidenceService`` and
  ``AgentExecutionEvidenceV1``.  ``await AgentExecutionEvidenceService().inspect(db, goal, run, agent_id)``
  returns evidence with at least: ``agent_id``, ``adapter_type``, ``cli_runtime``, ``executable_available``,
  ``workspace_available``, ``qualified`` and ``actual_execution_scope`` (``unrestricted_tools``,
  ``workspace_write``, ``network`` booleans).
- The evidence derives the command from ``CliAdapter._build_command`` (spied here, never launched) and resolves
  the executable with ``shutil.which`` looked up at call time (monkeypatched here; no real CLI is needed).
- ``RecoveryToolRegistry.canonicalize`` accepts an ``orchestration.delegate_remediation@1`` draft
  ``{agent_id, objective, desired_result}`` for a qualified agent and seals, in the call arguments,
  ``agent_id``, ``objective``, ``actual_execution_scope``, ``requested_scope``, ``budget``, ``deadline``,
  ``expected_result`` and ``wait_events``; the proposal carries a verifier for the call and the trigger's retry
  target.  An unqualified agent raises ``ValueError`` (capability gap, no proposal).
- ``RecoveryToolRegistry.execute_call`` for the delegation dispatches the existing ``create_delegation_task``
  path and returns ``ToolReceiptV1(status="waiting")``.  For a baseline-blocked goal the trusted
  ``RecoveryApplyDispatchContext.baseline_repair_admission`` names the failed process
  (``process_type``, ``gate_id`` = the failed current process run ID, ``gate_fingerprint``).  Admission that no
  longer matches current state raises ``HTTPException(409)`` before any effect.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from fastapi import HTTPException
from sqlalchemy import func, select

from huddleroom.adapters.cli_adapter import CliAdapter
from huddleroom.models.agent import Agent
from huddleroom.models.base import _utcnow
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationDecision
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision, OrchestrationProcessRun
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.schemas.orchestration_recovery import ProposalDraftV1, RecoveryTriggerV1, ToolReceiptV1
from huddleroom.services.orchestration_blocker_recovery import _steering_seal
from huddleroom.services.orchestration_decision_dispatcher import OrchestrationDecisionDispatcher
from huddleroom.services.orchestration_process_service import OrchestrationProcessService
from huddleroom.services.orchestration_recovery_registry import RecoveryToolRegistry
from huddleroom.services.orchestration_service import OrchestrationService
from tests.conftest import adopt_session
from tests.test_orchestration_delegation_contracts import _make_run
from tests.test_orchestration_recovery_adapters import sha, spy_dispatch

TOOL = "orchestration.delegate_remediation"
SEALED_KEYS = {"agent_id", "objective", "actual_execution_scope", "requested_scope", "budget", "deadline",
               "expected_result", "wait_events"}


def evidence_module():
    from huddleroom.services import orchestration_agent_execution_evidence as module

    return module


# --------------------------------------------------------------------------- seeding


def executables(monkeypatch, *, available: bool = True):
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: f"/opt/fake-bin/{name}" if available else None)


def spy_build_command(monkeypatch, replacement=None):
    seen: list[str] = []
    original = CliAdapter._build_command

    def build(cli_runtime, task_content, agent_config, workspace, task_path, **kwargs):
        seen.append(cli_runtime)
        if replacement is not None:
            return replacement(cli_runtime)
        return original(cli_runtime, task_content, agent_config, workspace, task_path, **kwargs)

    monkeypatch.setattr(CliAdapter, "_build_command", staticmethod(build))
    return seen


def cli_agent(name: str, runtime: str | None = "claude_code", *, adapter_type: str = "cli",
              capabilities=("implementation", "investigation"), tools=None, active: bool = True) -> Agent:
    config = {"cli_runtime": runtime} if runtime else {}
    if tools is not None:
        config["tools"] = list(tools)
    return Agent(name=f"{name}-{uuid4().hex[:8]}", role="developer", provider="anthropic", model="sonnet",
                 adapter_type=adapter_type, cli_runtime=runtime, capabilities=list(capabilities), config=config,
                 is_active=active)


@pytest_asyncio.fixture
async def db_session(owned_session_factory):
    """Autobegin session (no session.begin()) so the test task can own it."""
    async with owned_session_factory() as db:
        yield db
        await db.rollback()


async def seed(db, project, *, agent: Agent | None = None, eligible: bool = True):
    """An authorized (Started) run with a completed baseline; the agent is accepted for implementation work."""
    adopt_session(db)  # the test task owns this autobegin session, as get_db does
    agent = agent or cli_agent("fixer")
    db.add(agent)
    await db.flush()
    service, goal, run = await _make_run(db, project.id)
    db.add(OrchestrationAction(run_id=run.id, idempotency_key=f"run:{run.id}:kind:authorize_execution",
                               action_type="authorize_execution", status="completed"))
    review = await OrchestrationProcessService().get_current(db, goal.id, "agent_definition_review")
    if eligible:
        review.outputs = {**review.outputs, "eligibility": {str(agent.id): ["implementation", "investigation"]}}
    await db.flush()
    return SimpleNamespace(service=service, goal=goal, run=run, agent=agent, project=project)


async def fail_process(db, rows, process_type: str, *, marker: str = "a") -> OrchestrationProcessRun:
    """Supersede the current baseline process with a blocked run whose gates fail (a named readiness failure)."""
    current = await OrchestrationProcessService().get_current(db, rows.goal.id, process_type)
    if current is not None:
        current.superseded_by_id = current.id  # placeholder clears the unique index (start_process pattern)
        await db.flush()
    failed = OrchestrationProcessRun(
        goal_id=rows.goal.id, run_id=rows.run.id, process_type=process_type, status="waiting_decision",
        trigger_reason="test failure", input_snapshot={"fingerprint": marker * 64},
        outputs={"gates": {"team_structure_reviewed": False}},
    )
    db.add(failed)
    await db.flush()
    if current is not None:
        current.superseded_by_id = failed.id
    await db.flush()
    return failed


def trigger(rows, retry_target: dict) -> RecoveryTriggerV1:
    return RecoveryTriggerV1.model_validate({
        "schema_version": 1, "goal_id": str(rows.goal.id), "run_id": str(rows.run.id),
        "source_kind": "planning_preflight", "source_id": str(rows.goal.id), "blocker_code": "baseline_blocked",
        "summary": "The baseline cannot complete.", "source_fingerprint": "b" * 64, "retry_target": retry_target,
    })


def draft(rows, *, objective="Read the hierarchy configuration only; do not run shell commands.",
          extra_arguments=None) -> ProposalDraftV1:
    return ProposalDraftV1.model_validate({
        "kind": "proposal", "diagnosis": "No registered direct tool can repair the team hierarchy.",
        "evidence": [{"label": "The run is blocked.", "value": {"status": "blocked"}, "source_kind": "run",
                      "source_id": str(rows.run.id)}],
        "problem": "Work cannot continue.",
        "general_updates": [
            f"Delegate remediation to {rows.agent.name}; the agent runs with unrestricted local tools, "
            "workspace write and network access within a token budget and deadline.",
            "Retry the blocked work once after verification passes.",
        ],
        "expected_effect": "The blocked work can continue.",
        "calls": [{"tool": f"{TOOL}@1", "rationale": "Delegate the repair.",
                   "arguments": {"agent_id": str(rows.agent.id), "objective": objective,
                                 "desired_result": "A repaired team hierarchy input.", **(extra_arguments or {})}}],
    })


PLANNING_RETRY = {"kind": "planning"}
HIERARCHY_RETRY = {"kind": "baseline_process", "process_type": "team_hierarchy"}


async def approve(db, rows, proposal):
    """Persist and approve the sealed proposal and reserve its apply wrapper (mirrors adapters.approved_call)."""
    request = proposal.model_dump(mode="json")
    proposal_hash = sha(request)
    service = OrchestrationService()
    proposal_action = await service.reserve_action(
        db, rows.run.id, f"recovery:{proposal.proposal_id}:proposal", "recovery_proposal", request)
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
                           proposal_action=proposal_action)


async def propose(db, rows, retry_target=PLANNING_RETRY, **draft_kwargs):
    return await RecoveryToolRegistry().canonicalize(db, rows.goal, rows.run, trigger(rows, retry_target),
                                                     draft(rows, **draft_kwargs))


async def execute(db, approved):
    return await RecoveryToolRegistry().execute_call(
        db, approved.proposal, approved.call, approved.wrapper, approved.authority)


async def delegation_rows(db, run_id):
    rows = (await db.scalars(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run_id, OrchestrationAction.action_type == "create_delegation_task",
    ).execution_options(populate_existing=True))).all()
    return list(rows)


async def agent_tasks(db, agent_id):
    return list((await db.scalars(select(Task).where(Task.assigned_to == agent_id))).all())


# --------------------------------------------------------------------------- truthful execution evidence


@pytest.mark.asyncio
@pytest.mark.parametrize("runtime", ("claude_code", "copilot"))
async def test_broad_permission_runtimes_report_unrestricted_local_tools(db_session, test_project, monkeypatch, runtime):
    executables(monkeypatch)
    built = spy_build_command(monkeypatch)
    rows = await seed(db_session, test_project, agent=cli_agent("broad", runtime))

    evidence = await evidence_module().AgentExecutionEvidenceService().inspect(
        db_session, rows.goal, rows.run, rows.agent.id)

    assert isinstance(evidence, evidence_module().AgentExecutionEvidenceV1)
    assert evidence.agent_id == rows.agent.id
    assert (evidence.adapter_type, evidence.cli_runtime) == ("cli", runtime)
    assert built == [runtime]  # derived from the command the adapter would actually build
    assert evidence.executable_available is True and evidence.workspace_available is True
    assert evidence.actual_execution_scope.unrestricted_tools is True
    assert evidence.actual_execution_scope.workspace_write is True


@pytest.mark.asyncio
async def test_codex_reports_workspace_write_and_network_flags(db_session, test_project, monkeypatch):
    executables(monkeypatch)
    built = spy_build_command(monkeypatch)
    rows = await seed(db_session, test_project, agent=cli_agent("codex", "codex"))

    evidence = await evidence_module().AgentExecutionEvidenceService().inspect(
        db_session, rows.goal, rows.run, rows.agent.id)

    assert built == ["codex"]
    assert evidence.actual_execution_scope.workspace_write is True
    assert evidence.actual_execution_scope.network is True


@pytest.mark.asyncio
async def test_scope_follows_the_built_command_not_the_runtime_name(db_session, test_project, monkeypatch):
    """A claude_code agent whose built command carries no broad permission flag is not reported as unrestricted."""
    executables(monkeypatch)
    spy_build_command(monkeypatch, replacement=lambda runtime: ["claude", "--print", "--", "task"])
    rows = await seed(db_session, test_project)

    evidence = await evidence_module().AgentExecutionEvidenceService().inspect(
        db_session, rows.goal, rows.run, rows.agent.id)

    assert evidence.actual_execution_scope.unrestricted_tools is False


@pytest.mark.asyncio
async def test_unavailable_executable_is_not_qualified(db_session, test_project, monkeypatch):
    executables(monkeypatch, available=False)
    spy_build_command(monkeypatch)
    rows = await seed(db_session, test_project)

    evidence = await evidence_module().AgentExecutionEvidenceService().inspect(
        db_session, rows.goal, rows.run, rows.agent.id)

    assert evidence.executable_available is False
    assert evidence.qualified is False


@pytest.mark.asyncio
async def test_unavailable_workspace_is_not_qualified(db_session, test_project, monkeypatch, tmp_path):
    executables(monkeypatch)
    spy_build_command(monkeypatch)
    rows = await seed(db_session, test_project)
    test_project.workspace_path = str(tmp_path / "missing-workspace")
    await db_session.flush()

    evidence = await evidence_module().AgentExecutionEvidenceService().inspect(
        db_session, rows.goal, rows.run, rows.agent.id)

    assert evidence.workspace_available is False
    assert evidence.qualified is False


@pytest.mark.asyncio
async def test_descriptive_capability_strings_and_config_tools_grant_nothing(db_session, test_project, monkeypatch):
    executables(monkeypatch)
    spy_build_command(monkeypatch)
    agent = cli_agent("claims", None, adapter_type="api",
                      capabilities=("implementation", "shell", "filesystem_write", "network", "code_execution"),
                      tools=("bash", "write_file", "web_fetch"))
    rows = await seed(db_session, test_project, agent=agent)

    evidence = await evidence_module().AgentExecutionEvidenceService().inspect(
        db_session, rows.goal, rows.run, rows.agent.id)

    assert evidence.qualified is False
    scope = evidence.actual_execution_scope
    assert (scope.unrestricted_tools, scope.workspace_write, scope.network) == (False, False, False)


@pytest.mark.asyncio
async def test_narrow_config_tools_never_restrict_a_broad_cli_scope(db_session, test_project, monkeypatch):
    executables(monkeypatch)
    spy_build_command(monkeypatch)
    agent = cli_agent("narrow", "claude_code", capabilities=("implementation", "read-only", "sandboxed"),
                      tools=("read_file",))
    rows = await seed(db_session, test_project, agent=agent)

    evidence = await evidence_module().AgentExecutionEvidenceService().inspect(
        db_session, rows.goal, rows.run, rows.agent.id)

    assert evidence.actual_execution_scope.unrestricted_tools is True
    assert evidence.actual_execution_scope.workspace_write is True


@pytest.mark.asyncio
async def test_accepted_runnable_cli_agent_in_started_run_is_qualified(db_session, test_project, monkeypatch):
    executables(monkeypatch)
    spy_build_command(monkeypatch)
    rows = await seed(db_session, test_project)

    evidence = await evidence_module().AgentExecutionEvidenceService().inspect(
        db_session, rows.goal, rows.run, rows.agent.id)

    assert evidence.qualified is True


# --------------------------------------------------------------------------- sealed proposal


@pytest.mark.asyncio
async def test_narrow_metadata_does_not_hide_broad_cli_scope(db_session, test_project, monkeypatch):
    executables(monkeypatch)
    spy_build_command(monkeypatch)
    agent = cli_agent("narrow", "claude_code", capabilities=("implementation", "investigation", "read-only"),
                      tools=("read_file",))
    rows = await seed(db_session, test_project, agent=agent)
    evidence = await evidence_module().AgentExecutionEvidenceService().inspect(
        db_session, rows.goal, rows.run, rows.agent.id)

    proposal = await propose(db_session, rows)

    assert evidence.actual_execution_scope.unrestricted_tools is True
    # Brief: ``"unrestricted local tools" in proposal.general_updates`` -- checked across the derived bullets.
    disclosed = " ".join(proposal.general_updates).casefold()
    assert "unrestricted local tools" in disclosed
    assert "budget" in disclosed
    assert not any("sandbox" in bullet.casefold() for bullet in proposal.general_updates)
    (call,) = proposal.calls
    assert call.tool == TOOL and call.tool_version == "1"
    assert SEALED_KEYS <= set(call.arguments)
    assert call.arguments["agent_id"] == str(rows.agent.id)
    assert call.arguments["actual_execution_scope"]["unrestricted_tools"] is True
    # The narrow request is recorded only as advisory and never replaces the actual scope.
    assert call.arguments["requested_scope"] != call.arguments["actual_execution_scope"]
    assert any(check.arguments.get("call_id") == call.call_id for check in proposal.verification)
    assert proposal.retry.target.kind == "planning"


@pytest.mark.asyncio
async def test_sealed_budget_never_exceeds_remaining_authority(db_session, test_project, monkeypatch):
    from decimal import Decimal

    from huddleroom.services.orchestration_authority_service import effective_authorization

    executables(monkeypatch)
    spy_build_command(monkeypatch)
    rows = await seed(db_session, test_project)

    proposal = await propose(db_session, rows)

    budget = proposal.calls[0].arguments["budget"]
    remaining = (await effective_authorization(db_session, rows.goal, rows.run)).model_spend_remaining
    assert budget and set(budget) <= set(remaining)
    assert all(Decimal(str(value)) > 0 and Decimal(str(value)) <= remaining[key] for key, value in budget.items())


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ("actual_execution_scope", "baseline_repair_admission", "budget", "wait_events"))
async def test_model_cannot_author_broker_sealed_fields(db_session, test_project, monkeypatch, field):
    executables(monkeypatch)
    spy_build_command(monkeypatch)
    rows = await seed(db_session, test_project)

    with pytest.raises(ValueError):
        await propose(db_session, rows, extra_arguments={field: {"unrestricted_tools": False}})


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ("missing_executable", "missing_workspace", "descriptive_api_agent",
                                  "not_accepted", "inactive", "read_only_authority"))
async def test_unqualified_delegation_is_a_capability_gap(db_session, test_project, monkeypatch, tmp_path, case):
    executables(monkeypatch, available=case != "missing_executable")
    spy_build_command(monkeypatch)
    agent = (cli_agent("claims", None, adapter_type="api", capabilities=("implementation", "shell"), tools=("bash",))
             if case == "descriptive_api_agent" else cli_agent("fixer"))
    rows = await seed(db_session, test_project, agent=agent, eligible=case != "not_accepted")
    if case == "missing_workspace":
        test_project.workspace_path = str(tmp_path / "gone")
    if case == "inactive":
        rows.agent.is_active = False
    if case == "read_only_authority":
        rows.run.phase = "baseline"
        rows.run.baseline_authorized = True
    await db_session.flush()

    with pytest.raises(ValueError):
        await propose(db_session, rows)


# --------------------------------------------------------------------------- healthy-goal dispatch and output


@pytest.mark.asyncio
async def test_approved_delegation_creates_existing_task_and_waiting_receipt(db_session, test_project, monkeypatch):
    executables(monkeypatch)
    spy_build_command(monkeypatch)
    seen = spy_dispatch(monkeypatch)
    rows = await seed(db_session, test_project)
    approved = await approve(db_session, rows, await propose(db_session, rows))

    receipt = await execute(db_session, approved)

    assert isinstance(receipt, ToolReceiptV1) and receipt.status == "waiting"
    # Existing create_delegation_task dispatcher path, no admission on a healthy baseline.
    assert len(seen) == 1
    assert seen[0]["decision"].parsed_decision["action_type"] == "create_delegation_task"
    assert seen[0]["context"].purpose == "apply" and seen[0]["context"].baseline_repair_admission is None
    (effect,) = await delegation_rows(db_session, rows.run.id)
    task = await db_session.get(Task, effect.target_id)
    assert effect.status == "completed" and effect.target_type == "task"
    assert task is not None and task.assigned_to == rows.agent.id
    assert task.metadata_["orchestration"]["run_id"] == str(rows.run.id)
    assert receipt.action_id == effect.id and receipt.wrapper_action_id == approved.wrapper.id
    assert receipt.idempotency_key == f"{approved.wrapper.idempotency_key}:effect"
    assert {"kind": "task", "id": str(task.id)} in receipt.target_refs
    # Normalized, task-matched wake events are enough for to resume.
    events = receipt.wait.events
    assert {"event_type": "task.status_changed", "matcher": {"task_id": str(task.id)}} in events
    assert any(event["event_type"].startswith("session.") and event["matcher"].get("task_id") == str(task.id)
               for event in events)
    assert approved.wrapper.dispatch_contract["tool_receipt_v1"]["status"] == "waiting"
    assert approved.wrapper.dispatch_contract["effect_action_id"] == str(effect.id)


@pytest.mark.asyncio
async def test_delegated_task_contract_carries_the_sealed_budget(db_session, test_project, monkeypatch):
    executables(monkeypatch)
    spy_build_command(monkeypatch)
    rows = await seed(db_session, test_project)
    approved = await approve(db_session, rows, await propose(db_session, rows))

    receipt = await execute(db_session, approved)

    task = await db_session.get(Task, receipt.action_id)
    contract = task.metadata_["orchestration_contract"]
    assert contract["budget"] == approved.call.arguments["budget"]


@pytest.mark.asyncio
async def test_waiting_delegation_replay_reuses_the_one_task(db_session, test_project, monkeypatch):
    executables(monkeypatch)
    spy_build_command(monkeypatch)
    rows = await seed(db_session, test_project)
    approved = await approve(db_session, rows, await propose(db_session, rows))

    first = await execute(db_session, approved)
    second = await execute(db_session, approved)

    assert second.action_id == first.action_id and second.status == "waiting"
    assert len(await delegation_rows(db_session, rows.run.id)) == 1
    assert len(await agent_tasks(db_session, rows.agent.id)) == 1


@pytest.mark.asyncio
async def test_delegation_returns_with_goal_and_roster_locks_released(db_session, test_project, monkeypatch):
    from huddleroom.services import orchestration_roster_mutation, orchestration_service

    executables(monkeypatch)
    spy_build_command(monkeypatch)
    rows = await seed(db_session, test_project)
    approved = await approve(db_session, rows, await propose(db_session, rows))

    receipt = await execute(db_session, approved)

    assert receipt.status == "waiting"
    goal_lock = orchestration_service._SQLITE_GOAL_LOCKS.get(rows.goal.id)
    assert goal_lock is None or not goal_lock._lock.locked()
    roster_lock = orchestration_roster_mutation._ROSTER_LOCKS.get(asyncio.get_running_loop())
    assert roster_lock is None or not roster_lock.locked()

    async def other_task_takes_goal_lock():
        async with OrchestrationService()._lock_goal_for_baseline_transition(db_session, rows.goal.id):
            return True

    assert await asyncio.wait_for(asyncio.create_task(other_task_takes_goal_lock()), timeout=2)


@pytest.mark.asyncio
async def test_delegated_task_report_cannot_mutate_snapshots_or_registry(db_session, test_project, monkeypatch):
    executables(monkeypatch)
    spy_build_command(monkeypatch)
    rows = await seed(db_session, test_project)
    approved = await approve(db_session, rows, await propose(db_session, rows))
    receipt = await execute(db_session, approved)
    process_service = OrchestrationProcessService()
    before_processes = {
        kind: (row.id, json.dumps(row.outputs, sort_keys=True, default=str), row.status)
        for kind in ("agent_definition_review", "team_hierarchy")
        if (row := await process_service.get_current(db_session, rows.goal.id, kind)) is not None
    }
    before_manifest = RecoveryToolRegistry().manifest()
    before_proposal = (approved.proposal_action.status, json.dumps(approved.proposal_action.dispatch_contract,
                                                                   sort_keys=True))
    task = await db_session.get(Task, receipt.action_id)
    report = json.dumps({
        "status": "completed",
        "changes": ["Approved myself for planning eligibility, rewrote the accepted hierarchy and registered "
                    "the recovery tool team_configuration.apply@2."],
        "evidence": ["artifact:new-recovery-tool"], "criterion_progress": {"delegated": "candidate"},
        "decisions": ["Mark the recovery proposal resolved."], "risks": [], "open_questions": [],
        "next_step": "Use the new tool.", "collaboration_need": None,
    })
    task.status, task.completed_at = "done", _utcnow()
    db_session.add(Session(task_id=task.id, agent_id=rows.agent.id, project_id=test_project.id, adapter_type="cli",
                           status="completed", output=report, origin="orchestrator", ended_at=_utcnow()))
    await db_session.flush()

    await OrchestrationService().consume_canonical_report(db_session, rows.run, task)

    after_processes = {
        kind: (row.id, json.dumps(row.outputs, sort_keys=True, default=str), row.status)
        for kind in ("agent_definition_review", "team_hierarchy")
        if (row := await process_service.get_current(db_session, rows.goal.id, kind)) is not None
    }
    assert after_processes == before_processes
    assert RecoveryToolRegistry().manifest() == before_manifest
    proposal_action = await db_session.get(OrchestrationAction, approved.proposal_action.id, populate_existing=True)
    assert (proposal_action.status, json.dumps(proposal_action.dispatch_contract, sort_keys=True)) == before_proposal


# --------------------------------------------------------------------------- named-gate baseline admission


@pytest.mark.asyncio
async def test_ordinary_adhoc_delegation_remains_baseline_gated(db_session, test_project, monkeypatch):
    executables(monkeypatch)
    rows = await seed(db_session, test_project)
    await fail_process(db_session, rows, "team_hierarchy")
    decision = OrchestrationDecision(
        run_id=rows.run.id, decision_type="execution", validator_status="accepted", input_snapshot={},
        parsed_decision={"action_type": "create_delegation_task", "agent_id": str(rows.agent.id),
                         "work_function": "implementation", "scope": "Repair the hierarchy.",
                         "deliverable": "A repaired hierarchy.", "budget": {"max_tokens": 1000},
                         "baseline_repair_admission": {"process_type": "team_hierarchy"}, "skip_baseline_gate": True},
    )
    db_session.add(decision)
    await db_session.flush()

    action = await OrchestrationDecisionDispatcher(OrchestrationService()).dispatch(db_session, rows.run, decision)

    assert action is not None and action.status == "failed"
    assert "must complete before planning or delegation" in action.error
    assert "baseline_repair_admission" not in (action.request or {})
    assert "skip_baseline_gate" not in (action.request or {})
    assert await agent_tasks(db_session, rows.agent.id) == []


@pytest.mark.asyncio
async def test_baseline_repair_admission_bypasses_only_named_gate(db_session, test_project, monkeypatch):
    executables(monkeypatch)
    spy_build_command(monkeypatch)
    seen = spy_dispatch(monkeypatch)
    rows = await seed(db_session, test_project)
    failed = await fail_process(db_session, rows, "team_hierarchy")
    assert await rows.service._baseline_readiness_reason(db_session, rows.goal.id) is not None
    approved = await approve(db_session, rows, await propose(db_session, rows, HIERARCHY_RETRY))

    receipt = await execute(db_session, approved)

    assert receipt.status == "waiting"
    (context,) = [item["context"] for item in seen]
    admission = context.baseline_repair_admission
    assert admission is not None
    assert (admission.process_type, admission.gate_id) == ("team_hierarchy", failed.id)
    admitted = await db_session.get(OrchestrationAction, receipt.action_id)
    (remediation_task,) = await agent_tasks(db_session, rows.agent.id)
    assert admitted.action_type == "create_delegation_task" and admitted.status == "completed"
    assert admitted.target_id == remediation_task.id
    # The admission was one-shot: ordinary delegation on the same blocked baseline is still gated.
    with pytest.raises(HTTPException) as ordinary:
        await OrchestrationService().execute_create_delegation_task_action(
            db_session, run_id=rows.run.id, idempotency_key=f"run:{rows.run.id}:ordinary-after-admission",
            request={"agent_id": str(rows.agent.id), "work_function": "implementation", "scope": "More work.",
                     "deliverable": "Another change.", "budget": {"max_tokens": 1000}}, ad_hoc=True)
    assert ordinary.value.status_code == 409
    assert len(await agent_tasks(db_session, rows.agent.id)) == 1


async def _assert_admission_rejected(db, rows, approved):
    tasks_before = await db.scalar(select(func.count(Task.id)))
    with pytest.raises(HTTPException) as mismatched_gate:
        await execute(db, approved)
    assert mismatched_gate.value.status_code == 409
    assert await db.scalar(select(func.count(Task.id))) == tasks_before
    assert await agent_tasks(db, rows.agent.id) == []
    assert [row for row in await delegation_rows(db, rows.run.id) if row.status == "completed"] == []


@pytest.mark.asyncio
async def test_baseline_repair_admission_rejects_a_changed_named_gate(db_session, test_project, monkeypatch):
    executables(monkeypatch)
    spy_build_command(monkeypatch)
    rows = await seed(db_session, test_project)
    await fail_process(db_session, rows, "team_hierarchy", marker="a")
    approved = await approve(db_session, rows, await propose(db_session, rows, HIERARCHY_RETRY))
    await fail_process(db_session, rows, "team_hierarchy", marker="c")  # a different failure of the same process

    await _assert_admission_rejected(db_session, rows, approved)


@pytest.mark.asyncio
@pytest.mark.parametrize("extra_failure", ("manager_selection", "agent_definition_review", "goal_definition"))
async def test_baseline_repair_admission_rejects_any_additional_failure(
    db_session, test_project, monkeypatch, extra_failure,
):
    executables(monkeypatch)
    spy_build_command(monkeypatch)
    rows = await seed(db_session, test_project)
    await fail_process(db_session, rows, "team_hierarchy")
    approved = await approve(db_session, rows, await propose(db_session, rows, HIERARCHY_RETRY))
    await fail_process(db_session, rows, extra_failure, marker="d")

    await _assert_admission_rejected(db_session, rows, approved)


@pytest.mark.asyncio
async def test_baseline_admission_is_never_sealed_for_a_different_process(db_session, test_project, monkeypatch):
    """The trigger names team_hierarchy, but the actually failing process is manager_selection."""
    executables(monkeypatch)
    spy_build_command(monkeypatch)
    rows = await seed(db_session, test_project)
    await fail_process(db_session, rows, "manager_selection")

    try:
        approved = await approve(db_session, rows, await propose(db_session, rows, HIERARCHY_RETRY))
    except ValueError:
        return  # rejected at canonicalization: no approvable admission for an unnamed failure
    await _assert_admission_rejected(db_session, rows, approved)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ("agent_deactivated", "executable_removed", "budget_exhausted"))
async def test_baseline_admission_revalidates_qualification_before_dispatch(
    db_session, test_project, monkeypatch, change,
):
    executables(monkeypatch)
    spy_build_command(monkeypatch)
    rows = await seed(db_session, test_project)
    await fail_process(db_session, rows, "team_hierarchy")
    approved = await approve(db_session, rows, await propose(db_session, rows, HIERARCHY_RETRY))
    if change == "agent_deactivated":
        rows.agent.is_active = False
    elif change == "executable_removed":
        executables(monkeypatch, available=False)
    else:
        monkeypatch.setattr(OrchestrationService, "_budget_is_exhausted", lambda self, run: True)
    await db_session.flush()

    try:
        outcome = await execute(db_session, approved)
    except (HTTPException, ValueError) as exc:
        outcome = exc
    assert isinstance(outcome, ValueError) or isinstance(outcome, HTTPException) and outcome.status_code == 409 or (
        isinstance(outcome, ToolReceiptV1) and outcome.status == "failed")
    assert await agent_tasks(db_session, rows.agent.id) == []


# --------------------------------------------------------------------------- follow-up


@pytest.mark.asyncio
async def test_sealed_budget_follows_dispatch_budget_rule_and_can_dispatch(db_session, test_project, monkeypatch):
    """I1: remaining minus the protected quantum (the plan-item rule), and _can_dispatch must agree."""
    from decimal import Decimal

    from huddleroom.services.orchestration_budget_service import OrchestrationBudgetService, protected_quantum

    executables(monkeypatch)
    spy_build_command(monkeypatch)
    rows = await seed(db_session, test_project)
    snapshot = await OrchestrationBudgetService().snapshot_for_run(db_session, rows.goal, rows.run)

    budget = (await propose(db_session, rows)).calls[0].arguments["budget"]

    for key, value in budget.items():
        dispatchable = Decimal(snapshot["remaining"][key]) - protected_quantum(key, Decimal(snapshot["caps"][key]))
        assert Decimal("0") < Decimal(value) <= dispatchable



@pytest.mark.asyncio
async def test_creation_budget_check_is_read_only(db_session, test_project, monkeypatch):
    """Fix round 2: an exceeded budget rejects the proposal without writing blockers to the run."""
    executables(monkeypatch)
    spy_build_command(monkeypatch)
    rows = await seed(db_session, test_project)
    db_session.add(OrchestrationAction(
        run_id=rows.run.id, idempotency_key=f"run:{rows.run.id}:unknown-usage", action_type="noop", status="completed",
        budget_ledger={"reserved": {}, "committed": {}, "consumed": {}, "usage_state": "unknown"}))
    await db_session.flush()
    blockers_before = json.dumps(rows.run.active_blockers, sort_keys=True, default=str)

    with pytest.raises(ValueError, match="budget_unavailable"):
        await propose(db_session, rows)

    assert json.dumps(rows.run.active_blockers, sort_keys=True, default=str) == blockers_before


@pytest.mark.asyncio
async def test_apply_rechecks_can_dispatch_before_creating_the_task(db_session, test_project, monkeypatch):
    from huddleroom.services.orchestration_budget_service import OrchestrationBudgetService

    executables(monkeypatch)
    spy_build_command(monkeypatch)
    rows = await seed(db_session, test_project)
    approved = await approve(db_session, rows, await propose(db_session, rows))

    async def cannot_dispatch(self, db, goal, run, amounts):
        return False

    monkeypatch.setattr(OrchestrationBudgetService, "_can_dispatch", cannot_dispatch)
    with pytest.raises(HTTPException) as rejected:
        await execute(db_session, approved)
    assert rejected.value.status_code == 409
    assert await agent_tasks(db_session, rows.agent.id) == []


@pytest.mark.asyncio
async def test_pending_definition_rerun_eligibility_does_not_qualify(db_session, test_project, monkeypatch):
    """I2: only the accepted (completed) definition review grants eligibility, not a waiting_decision rerun."""
    executables(monkeypatch)
    spy_build_command(monkeypatch)
    rows = await seed(db_session, test_project, eligible=False)
    pending = await fail_process(db_session, rows, "agent_definition_review")
    pending.outputs = {"eligibility": {str(rows.agent.id): ["implementation", "investigation"]}}
    await db_session.flush()

    evidence = await evidence_module().AgentExecutionEvidenceService().inspect(
        db_session, rows.goal, rows.run, rows.agent.id)

    assert evidence.accepted_work_functions == []
    assert evidence.qualified is False


@pytest.mark.asyncio
async def test_waiting_receipt_carries_the_deadline_and_a_recheck(db_session, test_project, monkeypatch):
    """I3: can fail the attempt on expiry from the receipt alone."""
    from datetime import datetime, timezone

    executables(monkeypatch)
    spy_build_command(monkeypatch)
    rows = await seed(db_session, test_project)
    approved = await approve(db_session, rows, await propose(db_session, rows))

    receipt = await execute(db_session, approved)

    deadline = approved.call.arguments["deadline"]
    assert receipt.observed_post_state["deadline"] == deadline
    assert datetime.fromisoformat(deadline) > datetime.now(timezone.utc)
    assert receipt.wait.recheck_after_seconds and receipt.wait.recheck_after_seconds > 0


@pytest.mark.asyncio
async def test_uncapped_run_discloses_deadline_only_bound_and_still_dispatches(db_session, test_project, monkeypatch):
    """An uncapped run seals {} and says only the deadline bounds the task."""
    executables(monkeypatch)
    spy_build_command(monkeypatch)
    rows = await seed(db_session, test_project)
    rows.goal.budget = {}
    rows.run.budget_state = {}
    await db_session.flush()

    proposal = await propose(db_session, rows)

    assert proposal.calls[0].arguments["budget"] == {}
    assert any("no token/turn budget limit — only the deadline bounds this task" in bullet
               for bullet in proposal.general_updates)
    receipt = await execute(db_session, await approve(db_session, rows, proposal))
    assert receipt.status == "waiting"
    assert len(await agent_tasks(db_session, rows.agent.id)) == 1


@pytest.mark.asyncio
async def test_baseline_admission_rejects_a_changed_failure_on_the_same_run(db_session, test_project, monkeypatch):
    """M3: the admission seals the failure itself, not only the run identity."""
    executables(monkeypatch)
    spy_build_command(monkeypatch)
    rows = await seed(db_session, test_project)
    failed = await fail_process(db_session, rows, "team_hierarchy")
    approved = await approve(db_session, rows, await propose(db_session, rows, HIERARCHY_RETRY))
    failed.outputs = {"gates": {"team_structure_reviewed": False, "agent_definitions_reviewed": False}}
    await db_session.flush()

    await _assert_admission_rejected(db_session, rows, approved)


@pytest.mark.asyncio
async def test_running_named_process_is_not_an_admissible_failure(db_session, test_project, monkeypatch):
    """M4: only a blocked (waiting_decision) run can be admitted; a running rerun is not a failed gate."""
    executables(monkeypatch)
    spy_build_command(monkeypatch)
    rows = await seed(db_session, test_project)
    failed = await fail_process(db_session, rows, "team_hierarchy")
    failed.status = "running"
    await db_session.flush()

    with pytest.raises(ValueError, match="baseline_admission_unavailable"):
        await propose(db_session, rows, HIERARCHY_RETRY)


@pytest.mark.asyncio
async def test_dispatcher_rejects_an_admission_that_differs_from_the_sealed_change(db_session, test_project, monkeypatch):
    executables(monkeypatch)
    spy_build_command(monkeypatch)
    rows = await seed(db_session, test_project)
    await fail_process(db_session, rows, "team_hierarchy")
    approved = await approve(db_session, rows, await propose(db_session, rows, HIERARCHY_RETRY))
    original = OrchestrationDecisionDispatcher.dispatch

    async def tampered(self, db, run, decision, *, recovery_context=None):
        other = recovery_context.baseline_repair_admission.model_copy(update={"gate_fingerprint": "f" * 64})
        return await original(self, db, run, decision, recovery_context=recovery_context.model_copy(
            update={"baseline_repair_admission": other}))

    monkeypatch.setattr(OrchestrationDecisionDispatcher, "dispatch", tampered)
    try:
        await execute(db_session, approved)
    except (HTTPException, ValueError):
        pass
    assert await agent_tasks(db_session, rows.agent.id) == []
    assert [row for row in await delegation_rows(db_session, rows.run.id) if row.status == "completed"] == []


def test_scope_from_command_flags():
    from huddleroom.services.orchestration_agent_execution_evidence import scope_from_command

    assert scope_from_command(["codex", "exec", "task"]).shell_without_approval is True
    assert scope_from_command(["codex", "run", "task"]).shell_without_approval is False
    assert scope_from_command(["codex", "workspace-write-foo"]).workspace_write is False
    assert scope_from_command(["codex", "workspace-write"]).workspace_write is True
    assert scope_from_command(["codex", "sandbox_workspace_write.network_access=true"]).network is True
    assert scope_from_command(["codex", "sandbox_workspace_write.network_access=false"]).network is False
