"""Finite, code-owned recovery tool registry.

No model supplied name is ever resolved as a callable or service.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Annotated, Any, Literal
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from fastapi import HTTPException

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.agent import Agent
from huddleroom.models.artifact import Artifact
from huddleroom.models.graph import Graph, GraphRun
from huddleroom.models.meeting import Meeting
from huddleroom.models.session import Session
from huddleroom.models.orchestration import ACTIVE_RUN_STATUSES, OrchestrationAction, OrchestrationDecision, OrchestrationGate, OrchestrationGoal, OrchestrationRun, compute_goal_contract_fingerprint
from huddleroom.models.orchestration_process import OrchestrationAgentReview, OrchestrationAuthorityDecision, OrchestrationProcessRun, OrchestrationWarning
from huddleroom.models.task import Task
from huddleroom.schemas.orchestration_recovery import (
    ChangeV1, EvidenceV1, ProposalDraftV1, RecoveryCallV1, RecoveryCallDraftV1,
    RecoveryProposalV1, RecoveryTriggerV1, ToolManifestV1, ToolReceiptV1,
    VerificationCheckV1, VerificationResultV1, TeamConfigurationApplyV1, RecoveryApplyDispatchContext,
    BaselineRepairAdmissionV1,
)
from huddleroom.services.orchestration_agent_execution_evidence import (
    REMEDIATION_WORK_FUNCTIONS, AgentExecutionEvidenceService, AgentExecutionEvidenceV1,
)
from huddleroom.services.orchestration_recovery_configuration import RecoveryConfigurationService
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.orchestration_budget_service import OrchestrationBudgetService, protected_quantum
from huddleroom.services.orchestration_decision_dispatcher import OrchestrationDecisionDispatcher
from huddleroom.services.orchestration_steering import (
    OrchestrationSteeringService, SteeringVersionsChanged, active_direction_ids, steering_versions_from_snapshot,
)
from huddleroom.services.orchestration_agent_definition_analyzer import redact_semantic_payload
from huddleroom.services.orchestration_agent_definition_review import AgentDefinitionReviewProcess
from huddleroom.services.orchestration_goal_definition import goal_objective_with_clarifications
from huddleroom.services.orchestration_team_hierarchy import TeamHierarchyProcess, resolved_proposal_exclusions_from_decisions, select_plan_author
from huddleroom.services.orchestration_process_service import OrchestrationProcessService
from huddleroom.services.orchestration_linkage import check_goal_linkage
from huddleroom.services.orchestration_wake_when import normalize_wake_when

REGISTRY_VERSION = "1"
# Warning type of a recovery lineage blocker (written by the blocker-recovery coordinator).
RECOVERY_WARNING_TYPE = "recovery_blocked"
_TOOLS = (
    "team_configuration.apply", "orchestration.request_plan", "orchestration.request_plan_revision",
    "orchestration.request_roadmap_replan", "orchestration.retry_task", "orchestration.reassign_task",
    "orchestration.request_verification", "orchestration.schedule_meeting", "orchestration.start_graph",
    "orchestration.delegate_remediation",
)
_VERIFIERS = (
    "team_configuration.accepted", "planning.ready", "action.completed", "task.completed",
    "gate.satisfied", "process.completed", "warning.resolved", "artifact.exists",
)

class _Strict(BaseModel): model_config = ConfigDict(extra="forbid", frozen=True)
class _Plan(_Strict): agent_id: UUID; work_function: Literal["planning"]; scope: str = Field(min_length=1)
class _PlanRevision(_Strict): plan_task_id: UUID; revision_request: str = Field(min_length=1)
class _Roadmap(_Strict): agent_id: UUID; scope: str = Field(min_length=1); reason: str = Field(min_length=1)
class _Retry(_Strict): task_id: UUID; timeout: int | None = Field(None, gt=0); max_tokens: int | None = Field(None, gt=0)
class _Reassign(_Strict): task_id: UUID; agent_id: UUID
class _Verification(_Strict): gate_id: UUID; work_function: Literal["planning", "investigation", "implementation", "review", "validation", "summarization", "management"]
class _Meeting(_Strict):
    topic: str = Field(min_length=1); participant_agent_ids: list[UUID] = Field(min_length=1); task_id: UUID | None = None; gate_id: UUID | None = None; organizer_agent_id: UUID | None = None
class _Graph(_Strict): graph_id: UUID; subject_type: Literal["task", "artifact"]; subject_id: UUID
class _Delegation(_Strict): agent_id: UUID; objective: str = Field(min_length=1, max_length=4000); desired_result: str = Field(min_length=1, max_length=4000)
class _TeamAccepted(_Strict): goal_id: UUID; call_id: str = Field(min_length=1, max_length=100)
class _PlanningReady(_Strict): goal_id: UUID; run_id: UUID
class _ActionCompleted(_Strict): call_id: str = Field(min_length=1, max_length=100)
class _TaskResource(_Strict): source: Literal["resource"]; goal_id: UUID; run_id: UUID; task_id: UUID
class _TaskCall(_Strict): source: Literal["call"]; goal_id: UUID; run_id: UUID; call_id: str = Field(min_length=1, max_length=100)
class _GateSatisfied(_Strict): goal_id: UUID; run_id: UUID; gate_id: UUID
class _ProcessCompleted(_Strict): goal_id: UUID; process_id: UUID; expected_fingerprint: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
class _WarningResolved(_Strict): goal_id: UUID; warning_id: UUID
class _ArtifactResource(_Strict): source: Literal["resource"]; project_id: UUID; goal_id: UUID; artifact_id: UUID; artifact_type: str = Field(min_length=1); content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
class _ArtifactCall(_Strict): source: Literal["call"]; project_id: UUID; goal_id: UUID; call_id: str = Field(min_length=1, max_length=100); artifact_type: str = Field(min_length=1); content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
_TOOL_MODELS = {"team_configuration.apply": TeamConfigurationApplyV1, "orchestration.request_plan": _Plan, "orchestration.request_plan_revision": _PlanRevision, "orchestration.request_roadmap_replan": _Roadmap, "orchestration.retry_task": _Retry, "orchestration.reassign_task": _Reassign, "orchestration.request_verification": _Verification, "orchestration.schedule_meeting": _Meeting, "orchestration.start_graph": _Graph, "orchestration.delegate_remediation": _Delegation}
_VERIFIER_MODELS = {"team_configuration.accepted": _TeamAccepted, "planning.ready": _PlanningReady, "action.completed": _ActionCompleted, "task.completed": Annotated[_TaskResource | _TaskCall, Field(discriminator="source")], "gate.satisfied": _GateSatisfied, "process.completed": _ProcessCompleted, "warning.resolved": _WarningResolved, "artifact.exists": Annotated[_ArtifactResource | _ArtifactCall, Field(discriminator="source")]}
_ACTION_TYPES = {"orchestration.delegate_remediation": "create_delegation_task", "orchestration.request_plan": "request_plan", "orchestration.request_plan_revision": "request_plan_revision", "orchestration.request_roadmap_replan": "request_roadmap_replan", "orchestration.retry_task": "retry_task", "orchestration.reassign_task": "reassign_task", "orchestration.request_verification": "request_verification", "orchestration.schedule_meeting": "schedule_meeting", "orchestration.start_graph": "start_graph"}
_ACTION_LABELS = {name: action.replace("_", " ") for name, action in _ACTION_TYPES.items()}
_ACTION_LABELS["team_configuration.apply"] = "apply team configuration"
_ACTION_LABELS["orchestration.delegate_remediation"] = "delegate remediation"
_DELEGATION = "orchestration.delegate_remediation"
# ponytail: fixed policy constants; make them registry settings only if operators need to tune them.
_REMEDIATION_DEADLINE = timedelta(hours=4)
_REMEDIATION_BUDGET_SHARE = Decimal("0.5")  # at most half the remaining authority; the rest stays for the retry
# Events a delegated task produces; 's wait matches each on the created task's ID.
_REMEDIATION_WAIT_EVENTS = ("task.status_changed", "session.completed", "session.failed", "session.cancelled")
_SURFACE_LABELS = {
    "agent": "agent definitions", "eligibility": "eligibility", "assignments": "assignments",
    "manager": "manager", "reporting_hierarchy": "reporting hierarchy", "documented_gaps": "documented gaps",
}
_VERIFIER_LABELS = {name: name.replace(".", " ").replace("_", " ") for name in _VERIFIERS}


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _phrase(value: str) -> str:
    return " ".join(re.sub(r"[^\w:]+", " ", value.casefold()).split())


def _ordered_resources(value: Any, field: str = "") -> Any:
    if isinstance(value, dict):
        return {key: _ordered_resources(value[key], key) for key in sorted(value)}
    if isinstance(value, list):
        values = [_ordered_resources(item) for item in value]
        if field in {"capabilities", "target_ids", "review_ids", "accepted_target_ids", "accepted_review_ids",
                     "depends_on", "resolved_proposal_exclusions", "work_functions", "team_leads", "contributors",
                     "reviewers", "validators", "specialists", "assignments", "accepted_assignments",
                     "reporting_lines", "documented_gaps", "accepted_documented_gaps", "approved_for_work_functions"}:
            return sorted(values, key=lambda item: json.dumps(item, sort_keys=True))
        return values
    return value


@dataclass(frozen=True)
class ResourceSnapshotV1:
    values: dict[str, Any]
    fingerprint: str


class RecoveryToolRegistry:
    """The recovery control plane; registration is intentionally static."""

    def __init__(self, configuration_service: RecoveryConfigurationService | None = None) -> None:
        self.configuration_service = configuration_service or RecoveryConfigurationService()
        self.last_proposal_hash: str | None = None

    def manifest(self) -> tuple[ToolManifestV1, ...]:
        return tuple(
            ToolManifestV1(
                name=name, version="1",
                description=(f"Executable recovery tool: {name}. Allowed in proposal calls."
                             if name in _TOOL_MODELS else
                             f"Read-only verification predicate: {name}. Used only in server-owned verification checks; never in proposal calls."),
                arguments_schema=TypeAdapter(model).json_schema(),
            )
            for name, model in {**_TOOL_MODELS, **_VERIFIER_MODELS}.items()
        )

    @staticmethod
    def _selector(draft: RecoveryCallDraftV1) -> str:
        if draft.tool.count("@") != 1:
            raise ValueError("unknown_or_unversioned_tool")
        name, version = draft.tool.rsplit("@", 1)
        if version != "1" or name not in _TOOLS:
            raise ValueError("unknown_or_unversioned_tool")
        return name

    @staticmethod
    def _generic_changes(name: str, arguments: dict[str, Any], goal: OrchestrationGoal) -> tuple[ChangeV1, ...]:
        # Generic tools are sealed requests; their handlers are introduced by later tasks.
        target = arguments.get("task_id") or arguments.get("action_id") or arguments.get("agent_id") or str(goal.id)
        scope = "goal" if str(target) == str(goal.id) else "run"
        return (ChangeV1(operation="request", scope=scope, target_type=name, target_id=str(target), after=arguments),)

    async def _validate_scope(self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun,
                              name: str, arguments: dict[str, Any]) -> None:
        async def agent(value: str) -> None:
            row = await db.get(Agent, UUID(value))
            if row is None or not row.is_active: raise ValueError("out_of_scope_target")
        async def task(value: str) -> None:
            row = await db.get(Task, UUID(value))
            if row is None or row.project_id != goal.project_id or (row.metadata_ or {}).get("orchestration", {}).get("run_id") != str(run.id):
                raise ValueError("out_of_scope_target")
        if name in {"orchestration.request_plan", "orchestration.request_roadmap_replan", "orchestration.reassign_task"}: await agent(arguments["agent_id"])
        if name in {"orchestration.retry_task", "orchestration.reassign_task"}: await task(arguments["task_id"])
        if name == "orchestration.request_plan_revision": await task(arguments["plan_task_id"])
        if name == "orchestration.request_verification":
            gate = await db.get(OrchestrationGate, UUID(arguments["gate_id"]))
            if gate is None or gate.run_id != run.id: raise ValueError("out_of_scope_target")
        if name == "orchestration.schedule_meeting":
            for value in arguments["participant_agent_ids"] + ([arguments["organizer_agent_id"]] if arguments.get("organizer_agent_id") else []): await agent(value)
            if arguments.get("organizer_agent_id") and arguments["organizer_agent_id"] not in arguments["participant_agent_ids"]:
                raise ValueError("out_of_scope_target")
            service = OrchestrationService()
            try:
                with db.no_autoflush:
                    task_id = await service._meeting_task_id(db, run.id, arguments)
                    await service._coordination_task_and_gate(
                        db, run.id, task_id, UUID(arguments["gate_id"]) if arguments.get("gate_id") else None,
                    )
            except HTTPException:
                raise ValueError("out_of_scope_target") from None
        if name == "orchestration.start_graph":
            graph = await db.get(Graph, UUID(arguments["graph_id"]))
            if graph is None or not graph.is_active or graph.project_id not in {None, goal.project_id}:
                raise ValueError("out_of_scope_target")
            try:
                with db.no_autoflush:
                    await OrchestrationService()._graph_subject(db, run.id, arguments["subject_type"], UUID(arguments["subject_id"]))
            except HTTPException:
                raise ValueError("out_of_scope_target") from None

    async def _draft_summary_facts(self, db: AsyncSession, draft: ProposalDraftV1,
                                   calls: list[RecoveryCallV1]) -> tuple[set[str], list[set[str]]]:
        required = {_ACTION_LABELS[call.tool] for call in calls}
        identities: list[set[str]] = []
        for call, source in zip(calls, draft.calls):
            if call.tool == _DELEGATION:
                # The approver must read the actual runtime scope and the bound budget/deadline.
                scope = call.arguments["actual_execution_scope"]
                required.update(self._scope_phrases(scope) + ["budget", "deadline"])
                if scope.get("shell_without_approval") and not scope.get("unrestricted_tools"):
                    required.add("without approval")
                identities.append({(await db.get(Agent, UUID(call.arguments["agent_id"]))).name})
                continue
            if call.tool != "team_configuration.apply":
                continue
            required.update(_SURFACE_LABELS[change.target_type] for change in call.changes
                            if change.target_type in _SURFACE_LABELS)
            for item in source.arguments["agents"]:
                if item["operation"] == "create":
                    identities.append({item["ref"], item["definition"]["name"]})
                else:
                    agent = await db.get(Agent, UUID(str(item["agent_id"])))
                    identities.append({agent.name, item["set"].get("name") or agent.name})
            for gap in source.arguments.get("documented_gaps", []):
                required.add(gap["code"])
        return required, identities

    @staticmethod
    def _scope_phrases(scope: dict[str, Any]) -> list[str]:
        return [phrase for key, phrase in (("unrestricted_tools", "unrestricted local tools"),
                                           ("workspace_write", "workspace write"), ("network", "network"))
                if scope.get(key)]

    @staticmethod
    async def _remediation_budget(db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun) -> dict[str, str]:
        """Same rule as plan-item delegation: remaining minus the protected quantum, then a bounded share."""
        snapshot = await OrchestrationBudgetService().snapshot_for_run(db, goal, run)
        # Read-only at creation: _can_dispatch writes blockers, so it only runs again before apply.
        if any(Decimal(v) < 0 for v in snapshot["remaining"].values()) or (run.budget_state or {}).get("usage_state", "known") != "known":
            raise ValueError("agent_execution_unproven: budget_unavailable")
        budget = {}
        for key, value in snapshot["remaining"].items():
            amount = max(Decimal("0"), Decimal(value) - protected_quantum(key, Decimal(snapshot["caps"][key])))
            amount *= _REMEDIATION_BUDGET_SHARE
            amount = amount.to_integral_value(ROUND_DOWN) if key != "max_hours" else amount
            if amount <= 0:
                raise ValueError("agent_execution_unproven: budget_unavailable")
            budget[key] = format(amount.normalize(), "f")
        return budget

    @staticmethod
    def _summary(calls: list[RecoveryCallV1], checks: list[VerificationCheckV1],
                 retry_target: dict[str, Any], affected_goal_ids: set[str], agent_names: list[str],
                 call_bullets: dict[str, list[str]] | None = None) -> list[str]:
        bullets = []
        for call in calls:
            if call.call_id in (call_bullets or {}):
                bullets.extend(call_bullets[call.call_id])
                continue
            label = _ACTION_LABELS[call.tool].capitalize()
            if call.tool == "team_configuration.apply":
                surfaces = sorted({_SURFACE_LABELS[change.target_type] for change in call.changes
                                   if change.target_type in _SURFACE_LABELS})
                label += ": update " + ", ".join(surfaces)
                if agent_names:
                    label += " for " + ", ".join(sorted(set(agent_names)))
            bullets.append(label + ".")
        if any(change.scope == "global" for call in calls for change in call.changes):
            bullets.append(f"Change the global roster; refresh configuration for {len(affected_goal_ids)} affected goals.")
        gaps = sorted({f"{gap['code']}: {gap['rationale']}" for call in calls
                       if call.tool == "team_configuration.apply" for gap in call.arguments["documented_gaps"]})
        if gaps:
            bullets.append("Document gaps or overrides: " + "; ".join(gaps) + ".")
        bullets.append("Verify " + ", ".join(sorted({_VERIFIER_LABELS[check.verifier] for check in checks})) + ".")
        retry_label = retry_target["kind"].replace("_", " ")
        if retry_target["kind"] == "baseline_process":
            retry_label = retry_target["process_type"].replace("_", " ") + " baseline process"
        bullets.append("Retry " + retry_label + " once after verification passes.")
        return bullets

    @staticmethod
    def _checks(calls: list[RecoveryCallV1], goal: OrchestrationGoal, run: OrchestrationRun) -> list[VerificationCheckV1]:
        result: list[VerificationCheckV1] = []
        for index, call in enumerate(calls, 1):
            if call.tool == _DELEGATION:
                verifier = "task.completed"
                arguments = {"source": "call", "goal_id": str(goal.id), "run_id": str(run.id), "call_id": call.call_id}
            elif call.tool == "team_configuration.apply":
                verifier, arguments = "team_configuration.accepted", {"goal_id": str(goal.id), "call_id": call.call_id}
            else:
                verifier, arguments = "action.completed", {"call_id": call.call_id}
            TypeAdapter(_VERIFIER_MODELS[verifier]).validate_python(arguments)
            result.append(VerificationCheckV1(check_id=f"check-{index}", verifier=verifier, verifier_version="1", arguments=arguments))
        return result

    async def canonicalize(self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun,
                           trigger: RecoveryTriggerV1, draft: ProposalDraftV1) -> RecoveryProposalV1:
        if trigger.goal_id != goal.id or trigger.run_id != run.id:
            raise ValueError("trigger_out_of_scope")
        calls: list[RecoveryCallV1] = []
        call_bullets: dict[str, list[str]] = {}
        for index, call_draft in enumerate(draft.calls, 1):
            name = self._selector(call_draft)
            if name == "team_configuration.apply":
                prepared = await self.configuration_service.prepare(db, goal, run, TeamConfigurationApplyV1.model_validate(call_draft.arguments))
                arguments, changes = prepared.arguments, list(prepared.changes)
            elif name == _DELEGATION:
                arguments, changes, call_bullets[f"call-{index}"] = await self._prepare_delegation(
                    db, goal, run, trigger, _Delegation.model_validate(call_draft.arguments))
            else:
                model = _TOOL_MODELS[name].model_validate(call_draft.arguments)
                values = model.model_dump(mode="json")
                await self._validate_scope(db, goal, run, name, values)
                arguments = OrchestrationService().canonical_decision_request(_ACTION_TYPES[name], values, run_id=run.id)
                changes = list(self._generic_changes(name, arguments, goal))
            calls.append(RecoveryCallV1(call_id=f"call-{index}", tool=name, tool_version="1", arguments=arguments, changes=changes))
        if not calls:
            raise ValueError("missing_calls")
        checks = self._checks(calls, goal, run)
        if len(checks) > 8 or not checks:
            raise ValueError("missing_verifier")
        material = [change for call in calls for change in call.changes]
        required, identities = await self._draft_summary_facts(db, draft, calls)
        summary = " " + _phrase(" ".join(draft.general_updates)) + " "
        def covered(label: str) -> bool:
            return " " + _phrase(label) + " " in summary
        if any(not covered(label) for label in required) or any(not any(covered(label) for label in names) for names in identities):
            raise ValueError("summary_missing_material_effect")
        agent_names = []
        for call in calls:
            if call.tool != "team_configuration.apply":
                continue
            for item in call.arguments["agents"]:
                agent = await db.get(Agent, UUID(item["agent_id"])) if item["operation"] == "update" else None
                agent_names.append(item["definition"]["name"] if item["operation"] == "create" else item["set"].get("name") or agent.name)
        affected_goal_ids: set[str] = set()
        for change in material:
            if change.target_type != "configuration_fingerprint":
                continue
            affected_goal_ids.update(change.after if change.target_id == "aggregate" else [str(change.target_id)])
        general_updates = self._summary(calls, checks, trigger.retry_target.model_dump(mode="json"), affected_goal_ids,
                                        redact_semantic_payload(agent_names), call_bullets)
        proposal_id = uuid.uuid4()
        provisional = RecoveryProposalV1(schema_version=1, proposal_id=proposal_id, lineage_id=uuid.uuid4(), attempt=1,
            source={"kind": trigger.source_kind, "id": trigger.source_id, "blocker_code": trigger.blocker_code, "fingerprint": trigger.source_fingerprint},
            resource_fingerprint="0" * 64, registry_version=REGISTRY_VERSION, problem=draft.problem,
            general_updates=general_updates, expected_effect=draft.expected_effect, evidence=list(draft.evidence), calls=calls,
            verification=checks, retry={"target": trigger.retry_target.model_dump(mode="json"), "max_attempts": 1}, authority_required="human",
            global_effect=any(change.scope == "global" for change in material))
        resources = await self.capture_resources(db, goal, run, provisional)
        proposal = provisional.model_copy(update={"resource_fingerprint": resources.fingerprint})
        self.last_proposal_hash = _json_hash(proposal.model_dump(mode="json"))
        return proposal

    async def _prepare_delegation(self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun,
                                  trigger: RecoveryTriggerV1, draft: _Delegation) -> tuple[dict[str, Any], list[ChangeV1], list[str]]:
        """Seal one bounded remediation task from server evidence; the model chose only agent/objective/result."""
        evidence = await AgentExecutionEvidenceService().inspect(db, goal, run, draft.agent_id)
        if not evidence.qualified:
            raise ValueError("agent_execution_unproven: " + ", ".join(evidence.gaps))  # capability gap, not a proposal
        admission = await self._baseline_admission(db, goal, trigger)
        budget = await self._remediation_budget(db, goal, run)
        # Implementation is preferred: it is not an ad-hoc gated work function; investigation is the fallback.
        work_function = next(item for item in REMEDIATION_WORK_FUNCTIONS if item in evidence.accepted_work_functions)
        retry = trigger.retry_target.model_dump(mode="json")
        retry_label = (retry["process_type"].replace("_", " ") + " baseline process" if retry["kind"] == "baseline_process"
                       else retry["kind"].replace("_", " "))
        scope = self._actual_scope(evidence)
        deadline = (datetime.now(timezone.utc) + _REMEDIATION_DEADLINE).replace(microsecond=0).isoformat()
        values = {
            "agent_id": str(draft.agent_id), "work_function": work_function, "scope": draft.objective,
            "deliverable": draft.desired_result, "success_evidence": [draft.desired_result], "budget": budget,
            "title": "Recovery remediation", "decision_use": f"Unblock the blocked {retry_label} after verification.",
            "why_runnable": "An approved recovery fix delegates this one remediation.",
            "objective": draft.objective, "expected_result": draft.desired_result,
            # Advisory only: the runtime does not enforce it, so it never replaces the actual scope below.
            "requested_scope": {"description": draft.objective, "enforced": False},
            "actual_execution_scope": scope, "execution_evidence": evidence.model_dump(mode="json"),
            "deadline": deadline, "wait_events": [{"event_type": event, "match_on": "task_id"} for event in _REMEDIATION_WAIT_EVENTS],
            "verifier": "task.completed", "retry_target": retry,
        }
        try:
            arguments = OrchestrationService().canonical_decision_request("create_delegation_task", values, run_id=run.id)
        except HTTPException as exc:
            raise ValueError("invalid_delegation_arguments") from exc
        changes = [ChangeV1(operation="delegate", scope="run", target_type="delegation_task", target_id=str(draft.agent_id),
                            after={key: arguments[key] for key in ("agent_id", "work_function", "actual_execution_scope", "budget", "deadline")})]
        name = redact_semantic_payload([(await db.get(Agent, draft.agent_id)).name])[0]
        phrases = self._scope_phrases(scope)
        scope_text = ", ".join(phrases[:-1]) + (" and " if len(phrases) > 1 else "") + phrases[-1] if phrases else "no broad local tools"
        if scope["shell_without_approval"]:
            scope_text += ", and it runs shell commands without approval"
        budget_text = ", ".join(f"{key.removeprefix('max_').replace('_', ' ')} {value}" for key, value in sorted(budget.items()))
        bullets = [
            f"Delegate remediation to {name} as {work_function} work. Its runtime actually runs with {scope_text}; "
            "the requested task scope is advisory and not enforced.",
            f"Bound the task to one objective with a budget of {budget_text} and a deadline of {deadline}." if budget else
            f"Bound the task to one objective with no token/turn budget limit — only the deadline bounds this task ({deadline}).",
        ]
        if admission is not None:
            changes.append(ChangeV1(operation="request", scope="goal", target_type="baseline_repair_admission",
                                    target_id=str(admission.gate_id), after=admission.model_dump(mode="json")))
            bullets.append(f"Admit only this task past the blocked {admission.process_type.replace('_', ' ')} baseline process.")
        return arguments, changes, bullets

    @staticmethod
    def _actual_scope(evidence: AgentExecutionEvidenceV1) -> dict[str, Any]:
        return {**evidence.actual_execution_scope.model_dump(), "adapter_type": evidence.adapter_type,
                "cli_runtime": evidence.cli_runtime, "executable": evidence.executable}

    @staticmethod
    async def _baseline_admission(db: AsyncSession, goal: OrchestrationGoal,
                                  trigger: RecoveryTriggerV1) -> BaselineRepairAdmissionV1 | None:
        """None on a ready baseline; else admit only the trigger's one named blocked process."""
        service = OrchestrationService()
        if await service._baseline_processes_ready_for_goal(db, goal.id, allow_heal=False):
            return None
        target = trigger.retry_target
        current = (await OrchestrationProcessService().get_current(db, goal.id, target.process_type)
                   if target.kind == "baseline_process" else None)
        # Only a blocked (waiting_decision) named run is admissible. A running run is not a failure, and a
        # completed-but-stale process is not a named failed gate: both return baseline_admission_unavailable.
        if current is None or current.status != "waiting_decision":
            raise ValueError("baseline_admission_unavailable")
        admission = BaselineRepairAdmissionV1(process_type=target.process_type, gate_id=current.id,
                                              gate_fingerprint=service.baseline_gate_fingerprint(current))
        if not await service._baseline_processes_ready_for_goal(db, goal.id, allow_heal=False, admission=admission):
            raise ValueError("baseline_admission_unavailable")  # an additional baseline failure
        return admission

    async def _revalidate_delegation(self, db: AsyncSession, run: OrchestrationRun, call: RecoveryCallV1,
                                     admission: BaselineRepairAdmissionV1 | None) -> None:
        """Pre-effect revalidation (like the steering seal): any drift is a 409 and creates nothing."""
        goal = await db.get(OrchestrationGoal, run.goal_id)
        if not await OrchestrationService()._baseline_processes_ready_for_goal(db, goal.id, allow_heal=False, admission=admission):
            raise HTTPException(status_code=409, detail="baseline_repair_admission_rejected: the baseline changed since approval")
        arguments = call.arguments
        evidence = await AgentExecutionEvidenceService().inspect(db, goal, run, UUID(arguments["agent_id"]))
        budget_ok = not arguments["budget"] or await OrchestrationBudgetService()._can_dispatch(  # pylint: disable=protected-access
            db, goal, run, arguments["budget"])
        if (not evidence.qualified or self._actual_scope(evidence) != arguments["actual_execution_scope"]
                or arguments["work_function"] not in evidence.accepted_work_functions or not budget_ok):
            raise HTTPException(status_code=409, detail="agent_execution_unproven: " + (", ".join(evidence.gaps) or "scope_or_budget_changed"))
        if datetime.fromisoformat(arguments["deadline"]) <= datetime.now(timezone.utc):
            raise HTTPException(status_code=409, detail="remediation_deadline_passed")

    async def capture_resources(self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun,
                                proposal: RecoveryProposalV1) -> ResourceSnapshotV1:
        def row_fields(row: Any, *fields: str) -> dict[str, Any]:
            return {field: copy.deepcopy(getattr(row, field)) for field in fields}

        def task_value(task: Task) -> dict[str, Any]:
            value = row_fields(task, "id", "project_id", "title", "status", "assigned_to", "parent_id", "graph_run_id", "depends_on")
            metadata = (task.metadata_ or {}).get("orchestration", {})
            metadata = metadata if isinstance(metadata, dict) else {}
            value["orchestration"] = {key: metadata[key] for key in (
                "run_id", "goal_id", "work_function", "action_id", "authority_decision_id", "planning_version", "gate_id", "plan_item_gate_id",
            ) if key in metadata}
            plan_item = (task.metadata_ or {}).get("orchestration_plan_item", {})
            if isinstance(plan_item, dict) and "plan_item_gate_id" in plan_item:
                value["orchestration_plan_item"] = {"plan_item_gate_id": plan_item["plan_item_gate_id"]}
            return value

        def process_value(process: OrchestrationProcessRun) -> dict[str, Any]:
            outputs = process.outputs if isinstance(process.outputs, dict) else {}
            value = row_fields(process, "id", "goal_id", "run_id", "process_type", "status", "superseded_by_id")
            value["fingerprint"] = (process.input_snapshot or {}).get("fingerprint")
            for key in ("fingerprint", "coverage_fingerprint", "hierarchy_fingerprint", "target_ids", "review_ids",
                        "source_agent_review_process_id", "eligibility", "assignments", "role_to_agent", "hierarchy",
                        "documented_gaps", "approval", "approval_decision_id"):
                if key in outputs:
                    value["accepted_" + key] = copy.deepcopy(outputs[key])
            provenance = outputs.get("recovery_provenance")
            if isinstance(provenance, dict):
                value["recovery_provenance"] = {key: provenance[key] for key in (
                    "proposal_id", "proposal_hash", "call_id", "canonical_request_hash",
                ) if key in provenance}
            if isinstance(outputs.get("analysis"), dict):
                value["analysis_fingerprint"] = _json_hash(_ordered_resources(RecoveryConfigurationService._disclosure_value(outputs["analysis"])))
            return value

        references: dict[str, set[UUID]] = {key: set() for key in ("agent", "task", "gate", "graph", "artifact", "action")}
        fields_by_tool = {
            "orchestration.request_plan": {"agent_id": "agent"},
            "orchestration.request_plan_revision": {"plan_task_id": "task"},
            "orchestration.request_roadmap_replan": {"agent_id": "agent"},
            "orchestration.retry_task": {"task_id": "task"},
            "orchestration.reassign_task": {"task_id": "task", "agent_id": "agent"},
            "orchestration.request_verification": {"gate_id": "gate"},
            "orchestration.schedule_meeting": {"organizer_agent_id": "agent", "task_id": "task", "gate_id": "gate"},
            "orchestration.start_graph": {"graph_id": "graph"},
            "orchestration.delegate_remediation": {"agent_id": "agent"},
        }
        for call in proposal.calls:
            arguments = call.arguments
            for key, kind in fields_by_tool.get(call.tool, {}).items():
                if arguments.get(key):
                    references[kind].add(UUID(str(arguments[key])))
            if call.tool == "orchestration.schedule_meeting":
                references["agent"].update(UUID(str(value)) for value in arguments.get("participant_agent_ids", []))
            if call.tool == "orchestration.start_graph":
                references[arguments["subject_type"]].add(UUID(str(arguments["subject_id"])))
        retry = proposal.retry.target
        if retry.kind == "action":
            references["action"].add(retry.source_action_id)

        with db.no_autoflush:
            service = OrchestrationService()
            for call in proposal.calls:
                try:
                    if call.tool == "orchestration.schedule_meeting":
                        task_id = await service._meeting_task_id(db, run.id, call.arguments)
                        _, task, gate = await service._coordination_task_and_gate(
                            db, run.id, task_id, UUID(call.arguments["gate_id"]) if call.arguments.get("gate_id") else None,
                        )
                    elif call.tool == "orchestration.start_graph":
                        _, task, gate, _ = await service._graph_subject(
                            db, run.id, call.arguments["subject_type"], UUID(call.arguments["subject_id"]),
                        )
                    else:
                        continue
                except HTTPException:
                    raise ValueError("out_of_scope_target") from None
                references["task"].add(task.id)
                references["gate"].add(gate.id)
            runs = list((await db.execute(select(OrchestrationRun).where(
                OrchestrationRun.status.in_(ACTIVE_RUN_STATUSES)
            ))).scalars()) if proposal.global_effect else [run]
            runs_by_id = {item.id: item for item in runs}; runs_by_id[run.id] = run
            goal_ids = {item.goal_id for item in runs_by_id.values()}
            goals = list((await db.execute(select(OrchestrationGoal).where(OrchestrationGoal.id.in_(goal_ids)))).scalars())
            processes = list((await db.execute(select(OrchestrationProcessRun).where(
                OrchestrationProcessRun.goal_id.in_(goal_ids), OrchestrationProcessRun.superseded_by_id.is_(None),
                OrchestrationProcessRun.process_type.in_(("agent_definition_review", "team_hierarchy")),
            ))).scalars())
            trivial_ids = {item.id for item in goals if item.weight == "trivial"}
            review_ids_by_goal = {process.goal_id: TeamHierarchyProcess._review_ids(process.outputs or {})
                                 for process in processes if process.goal_id in trivial_ids
                                 and process.process_type == "agent_definition_review"} if proposal.global_effect else {}
            review_ids = {identifier for identifiers in review_ids_by_goal.values() for identifier in identifiers}
            reviews = list((await db.execute(select(OrchestrationAgentReview).where(
                OrchestrationAgentReview.id.in_(review_ids),
            ))).scalars()) if review_ids else []
            decisions = list((await db.execute(select(OrchestrationAuthorityDecision).where(
                OrchestrationAuthorityDecision.goal_id.in_(goal_ids),
            ))).scalars()) if proposal.global_effect else []
            authority_actions = list((await db.execute(select(OrchestrationAction).where(
                OrchestrationAction.run_id.in_(runs_by_id),
                OrchestrationAction.idempotency_key.startswith("authority_decision_delegate:"),
            ))).scalars()) if proposal.global_effect else []
            project_ids = {item.project_id for item in goals}
            project_tasks = list((await db.execute(select(Task).where(Task.project_id.in_(project_ids)))).scalars()) if proposal.global_effect else []
            agents = list((await db.execute(select(Agent).where(Agent.is_active.is_(True)))).scalars()) if proposal.global_effect else []
            if proposal.global_effect:
                by_id = {agent.id: agent for agent in agents}
                by_goal = {item.id: item for item in goals}
                definition_process = AgentDefinitionReviewProcess()
                for item in runs_by_id.values():
                    project_id = by_goal[item.goal_id].project_id
                    for _, agent, _ in await definition_process._current_run_assignments(db, project_id, item.id):
                        by_id[agent.id] = agent
                for item in goals:
                    if item.manager_agent_id is not None:
                        manager = await db.get(Agent, item.manager_agent_id)
                        if manager is not None:
                            by_id[manager.id] = manager
                agents = list(by_id.values())
            values: dict[str, Any] = {"registry": {"version": REGISTRY_VERSION}}
            for item in goals:
                values[f"goal_contract:{item.id}"] = {
                    **row_fields(item, "id", "project_id", "status", "weight", "authority_model", "manager_agent_id", "manager_user_id"),
                    "fingerprint": compute_goal_contract_fingerprint(item),
                    "objective_with_clarifications": goal_objective_with_clarifications(item),
                }
            # Recovery's own lineage blocker warnings mirror into active_blockers; they are recovery bookkeeping,
            # not repair state, so one lineage's warning never stales another lineage's proposal.
            recovery_warning_ids = {str(identifier) for identifier in (await db.execute(select(OrchestrationWarning.id).where(
                OrchestrationWarning.goal_id.in_(goal_ids), OrchestrationWarning.warning_type == RECOVERY_WARNING_TYPE,
            ))).scalars()}
            for item in runs_by_id.values():
                value = row_fields(item, "id", "goal_id", "status", "phase", "plan_state", "retry_state", "active_blockers")
                if isinstance(value["active_blockers"], list):
                    value["active_blockers"] = [entry for entry in value["active_blockers"] if not (
                        isinstance(entry, dict) and str(entry.get("warning_id")) in recovery_warning_ids)]
                values[f"run:{item.id}"] = value
            for process in processes:
                values[f"process:{process.process_type}:{process.id}"] = process_value(process)
            for review in reviews:
                if review.agent_id is not None and review.id in review_ids_by_goal.get(review.goal_id, set()):
                    values[f"review:{review.id}"] = row_fields(review, "id", "goal_id", "run_id", "agent_id",
                        "source_process_run_id", "approved_for_work_functions")
            run_ids = {str(identifier) for identifier in runs_by_id}
            for task in project_tasks:
                metadata = (task.metadata_ or {}).get("orchestration", {})
                if isinstance(metadata, dict) and metadata.get("run_id") in run_ids:
                    values[f"task:{task.id}"] = task_value(task)
            for decision in decisions:
                # Recovery approvals are the authority over this state, not part of it: a proposal's
                # own approve_fix card (created after sealing, answered later) must not stale it.
                if decision.decision_key.startswith("recovery:"):
                    continue
                exclusion = (decision.status == "answered" and decision.decision_key.startswith("team_hierarchy:agent:")
                             and decision.selected_option in {"edit", "reject"})
                if not exclusion and (decision.related_action_id is None or decision.run_id not in runs_by_id):
                    continue
                value = row_fields(decision, "id", "goal_id", "run_id", "decision_key", "status", "selected_option", "authority",
                                   "source_process_run_id", "related_action_id", "authority_agent_id", "decided_by_agent_id", "decided_by_user_id")
                if exclusion:
                    value["resolved_proposal_exclusions"] = [
                        {"action": item["action"], "definition_fingerprint": _json_hash(_ordered_resources(RecoveryConfigurationService._disclosure_value(item["definition"])))}
                        for item in resolved_proposal_exclusions_from_decisions([decision])
                    ]
                values[f"authority_decision:{decision.id}"] = value
            for action in authority_actions:
                values[f"action:{action.id}"] = row_fields(action, "id", "run_id", "action_type", "status", "idempotency_key")
            for agent in agents:
                values[f"agent:{agent.id}"] = row_fields(agent, "id", "name", "role", "description", "provider", "model", "system_prompt",
                    "adapter_type", "cli_runtime", "capabilities", "config", "is_active")
            models = {"agent": Agent, "task": Task, "gate": OrchestrationGate, "graph": Graph, "artifact": Artifact, "action": OrchestrationAction}
            for kind, identifiers in references.items():
                for identifier in sorted(identifiers, key=str):
                    row = await db.get(models[kind], identifier)
                    if row is None:
                        raise ValueError("out_of_scope_target")
                    if kind in {"task", "artifact"} and row.project_id != goal.project_id:
                        raise ValueError("out_of_scope_target")
                    if kind == "agent":
                        value = row_fields(row, "id", "name", "role", "description", "provider", "model", "system_prompt",
                                           "adapter_type", "cli_runtime", "capabilities", "config", "is_active")
                    elif kind == "task":
                        value = task_value(row)
                    elif kind == "gate":
                        value = row_fields(row, "id", "run_id", "success_criterion_key", "gate_type", "required_evidence", "status", "failure_reason")
                    elif kind == "graph":
                        value = row_fields(row, "id", "project_id", "name", "version", "is_active", "definition", "triggers", "escalation_chain")
                    elif kind == "artifact":
                        value = row_fields(row, "id", "project_id", "artifact_type", "status", "content_hash", "version", "linked_task_id")
                    else:
                        if row.run_id != run.id:
                            raise ValueError("out_of_scope_target")
                        value = row_fields(row, "id", "run_id", "action_type", "status", "target_type", "target_id", "idempotency_key")
                        value["request_fingerprint"] = _json_hash(RecoveryConfigurationService._disclosure_value(row.request))
                    values[f"{kind}:{identifier}"] = value
        ordered = _ordered_resources(RecoveryConfigurationService._disclosure_value(values))
        return ResourceSnapshotV1(ordered, _json_hash(ordered))

    async def execute_call(self, db: AsyncSession, proposal: RecoveryProposalV1, call: RecoveryCallV1,
                           wrapper: OrchestrationAction, authority_decision: OrchestrationAuthorityDecision) -> ToolReceiptV1:
        if call.tool == "team_configuration.apply":
            # Database aggregate: executes it inside its wrapper; it never enters the dispatcher.
            return await self.configuration_service.apply(db, proposal, call, wrapper, authority_decision)
        action_type = _ACTION_TYPES[call.tool]
        delegation = call.tool == _DELEGATION
        admissions = [BaselineRepairAdmissionV1.model_validate(change.after) for change in call.changes
                      if change.target_type == "baseline_repair_admission"]
        admission = admissions[0] if delegation and admissions else None
        wrapper_key = f"recovery:{proposal.proposal_id}:apply:{call.call_id}"
        if self._verification_call(proposal, call.call_id) != call or wrapper.idempotency_key != wrapper_key:
            raise ValueError("call_differs_from_sealed_proposal")
        stored = (wrapper.dispatch_contract or {}).get("tool_receipt_v1")
        if stored is not None:
            return ToolReceiptV1.model_validate(stored)  # replay of a finished call: the one linked effect
        run = await db.get(OrchestrationRun, wrapper.run_id)
        context = RecoveryApplyDispatchContext(
            purpose="apply", proposal_id=proposal.proposal_id, call_id=call.call_id,
            authority_decision_id=authority_decision.id, baseline_repair_admission=admission,
        )
        service = OrchestrationService()
        dispatcher = OrchestrationDecisionDispatcher(service)
        request = service.canonical_decision_request(action_type, call.arguments, run_id=run.id)
        effect_key = f"{wrapper_key}:effect"
        linked_id = (wrapper.dispatch_contract or {}).get("effect_action_id")
        if linked_id is not None:
            # Replay finds the one linked effect and never re-dispatches it (no blind replay).
            effect = await db.get(OrchestrationAction, UUID(str(linked_id)))
            if effect is None or effect.run_id != run.id or effect.idempotency_key != effect_key:
                raise ValueError("recovery_effect_link_mismatch")
        else:
            await self._assert_sealed_steering(db, run, authority_decision)
            if delegation:
                await self._revalidate_delegation(db, run, call, admission)
            input_snapshot = {}
            if action_type == "request_plan":
                run = await db.get(OrchestrationRun, run.id, populate_existing=True)
                input_snapshot = {"run": {"id": str(run.id), "plan_state": copy.deepcopy(run.plan_state)}}
            decision = OrchestrationDecision(
                run_id=run.id, decision_type="recovery_apply", validator_status="accepted",
                # Steering is fenced against the versions sealed at proposal time (_assert_sealed_steering),
                # not via input_snapshot: the ordinary fence would re-key the approved ``:effect`` row.
                input_snapshot=input_snapshot,
                parsed_decision={**call.arguments, "action_type": action_type, "applies_decision_id": str(authority_decision.id)},
            )
            db.add(decision)
            await db.flush()
            try:
                # The dispatcher hands the context's admission to the executor only after validating approve_fix.
                effect = await dispatcher.dispatch(db, run, decision, recovery_context=context)
            except HTTPException as exc:
                # Executors that own their 4xx state (roadmap replan) keep the failed row and raise.
                effect = await service._failed_action_for_key(db, run.id, effect_key)  # pylint: disable=protected-access
                if effect is None or not service._is_genuine_rejection(exc):  # pylint: disable=protected-access
                    raise
                dispatcher.link_recovery_effect(wrapper, context, request, effect)
            if effect is None:
                raise ValueError(decision.rejection_reason or "recovery_dispatch_rejected")
        # A created remediation task is awaited, not applied: resumes on its task/session events.
        status = {"completed": "waiting" if delegation else "applied", "failed": "failed"}.get(effect.status, "effect_unknown")
        target = ({"kind": effect.target_type, "id": str(effect.target_id)} if effect.target_id
                  else {"kind": "action", "id": str(effect.id)})
        error = (effect.error or "recovery_effect_failed") if status == "failed" else None
        receipt = ToolReceiptV1.model_validate({
            "proposal_id": str(proposal.proposal_id), "call_id": call.call_id, "tool": f"{call.tool}@{call.tool_version}",
            "wrapper_action_id": str(wrapper.id), "action_id": str(effect.id), "idempotency_key": effect.idempotency_key,
            "canonical_request_hash": dispatcher._canonical_request_hash(request),  # pylint: disable=protected-access
            "execution_identity": {"proposal_id": str(proposal.proposal_id), "call_id": call.call_id,
                                   "authority_decision_id": str(authority_decision.id), "purpose": "apply"},
            "status": status, "target_refs": [target],
            "observed_post_state": {"action_id": str(effect.id), "action_type": effect.action_type,
                                    "status": effect.status, **({"target": target} if effect.target_id else {}),
                                    **({"deadline": call.arguments["deadline"]} if delegation else {})},
            "evidence": [{"label": f"Recovery effect {action_type.replace('_', ' ')} is {effect.status}.",
                          "value": {"action_type": effect.action_type, "status": effect.status},
                          "source_kind": "action", "source_id": str(effect.id)}],
            "error_code": error[:255] if error else None, "error": error[:4000] if error else None,
            # The deadline recheck lets fail the attempt on expiry (absolute deadline is in observed_post_state).
            "wait": {"events": [{"event_type": event, "matcher": {"task_id": target["id"]}} for event in _REMEDIATION_WAIT_EVENTS],
                     "recheck_after_seconds": max(1, int((datetime.fromisoformat(call.arguments["deadline"])
                                                          - datetime.now(timezone.utc)).total_seconds())),
                     "expected_result": call.arguments["expected_result"]} if status == "waiting" else None,
        })
        if status == "effect_unknown":
            return receipt  # still-reserved effect: not replayed; failing both rows is 's reconciliation
        wrapper.dispatch_contract = {**(wrapper.dispatch_contract or {}), "tool_receipt_v1": receipt.model_dump(mode="json")}
        if status in {"applied", "waiting"}:
            await service._mark_action_completed(db, wrapper, target["kind"], UUID(target["id"]))  # pylint: disable=protected-access
        else:
            await service._fail_reserved_action_in_transaction(db, wrapper, error)  # pylint: disable=protected-access
        return receipt

    @staticmethod
    async def _assert_sealed_steering(db: AsyncSession, run: OrchestrationRun,
                                      authority_decision: OrchestrationAuthorityDecision) -> None:
        """Fail closed: raise SteeringVersionsChanged unless the steering sealed at proposal time still holds.

        ``{}`` is a valid seal only while steering yields no context (disabled); a missing or malformed
        seal never passes."""
        proposal_action = await db.get(OrchestrationAction, authority_decision.related_action_id)
        contract = (proposal_action.dispatch_contract or {}) if proposal_action else {}
        if not isinstance(contract.get("steering"), dict):
            raise SteeringVersionsChanged("recovery proposal has no steering seal")
        sealed = contract["steering"]
        goal = await db.get(OrchestrationGoal, run.goal_id)
        await db.flush()
        steering = OrchestrationSteeringService()
        if not sealed:
            if await steering.context_snapshot(db, goal, run, fresh=True):
                raise SteeringVersionsChanged("steering became active after approval")
            return
        versions = steering_versions_from_snapshot({"steering": sealed})
        directions = sealed.get("active_directions")
        expected_ids = active_direction_ids({"steering": sealed})
        if versions is None or not isinstance(directions, list) or len(expected_ids) != len(directions):
            raise SteeringVersionsChanged("recovery steering seal is invalid")
        # ponytail: unlocked check; take the goal lock like _steering_action_fence if a race shows up
        await steering.assert_current_snapshot(db, goal, run, versions, expected_ids)

    async def verify(self, db: AsyncSession, proposal: RecoveryProposalV1,
                     check: VerificationCheckV1) -> VerificationResultV1:
        if check.verifier not in _VERIFIERS or check.verifier_version != "1":
            raise ValueError("unknown_or_unversioned_verifier")
        try:
            arguments = TypeAdapter(_VERIFIER_MODELS[check.verifier]).validate_python(check.arguments)
        except ValueError as exc:
            raise ValueError("invalid_verifier_arguments") from exc
        # Every selector below is read-only, including calls that would ordinarily heal stale baseline inputs.
        with db.no_autoflush:
            origin = await self._verification_origin(db, proposal)
            evidence: list[EvidenceV1] = []
            passed, wake = False, None
            try:
                goal, run = (origin[1], origin[2]) if origin else (None, None)
                in_goal = goal is not None and getattr(arguments, "goal_id", goal.id) == goal.id
                in_run = run is not None and getattr(arguments, "run_id", run.id) == run.id
                if check.verifier == "team_configuration.accepted":
                    rows = [await OrchestrationProcessService().get_current(db, arguments.goal_id, kind)
                            for kind in ("agent_definition_review", "team_hierarchy")]
                    evidence.extend(self._verification_evidence(check, "process", row) for row in rows if row)
                    call = self._verification_call(proposal, arguments.call_id)
                    if in_goal and call is not None and call.tool == "team_configuration.apply":
                        passed = await self._team_configuration_accepted(db, proposal, call, goal, run, rows)
                elif check.verifier == "planning.ready":
                    row = await db.get(OrchestrationRun, arguments.run_id)
                    evidence.append(self._verification_evidence(check, "run", row, arguments.run_id))
                    if in_goal and in_run and run.status in ACTIVE_RUN_STATUSES:
                        service = OrchestrationService()
                        if await service._baseline_processes_ready_for_goal(db, goal.id, allow_heal=False):
                            author_id = await select_plan_author(db, goal, run)
                            definition = await OrchestrationProcessService().get_current(db, goal.id, "agent_definition_review")
                            hierarchy = await OrchestrationProcessService().get_current(db, goal.id, "team_hierarchy")
                            eligibility = (definition.outputs or {}).get("eligibility", {}) if definition else {}
                            author = await db.get(Agent, author_id) if author_id else None
                            if author and author.is_active and "planning" in eligibility.get(str(author.id), []):
                                await service._ensure_plan_request_allowed(db, run, check_readiness=False)
                                passed = self._configuration_gates(definition, hierarchy)
                elif check.verifier in {"action.completed", "task.completed", "artifact.exists"}:
                    if check.verifier == "task.completed" and arguments.source == "resource":
                        row = await db.get(Task, arguments.task_id)
                        evidence.append(self._verification_evidence(check, "task", row, arguments.task_id))
                        passed = in_goal and in_run and self._task_in_run(row, goal, run) and row.status == "done"
                    else:
                        calls = ([self._verification_call(proposal, arguments.call_id)]
                                 if hasattr(arguments, "call_id") else list(proposal.calls))
                        matches = []
                        for call in calls:
                            if call is None:
                                continue
                            wrapper = await self._verification_wrapper(db, proposal, call, run)
                            if wrapper:
                                evidence.append(self._verification_evidence(check, "action", wrapper))
                            bound = await self._bound_receipt(db, proposal, call, wrapper, origin) if origin else None
                            if bound is not None:
                                matches.append(bound)
                        if check.verifier == "action.completed" and len(matches) == 1:
                            receipt, effect = matches[0]
                            passed = receipt.status == "applied"
                            wake = await self._receipt_live_wait(db, receipt, goal, run) if receipt.status == "waiting" else None
                            evidence.append(self._verification_evidence(check, "action", effect))
                        elif check.verifier == "task.completed" and in_goal and in_run and len(matches) == 1:
                            receipt, _effect = matches[0]
                            task_refs = [ref for ref in receipt.target_refs if ref["kind"] == "task"]
                            if len(task_refs) == 1:
                                row = await db.get(Task, UUID(task_refs[0]["id"]))
                                evidence.append(self._verification_evidence(check, "task", row))
                                # A waiting receipt awaited exactly this task, so its completion satisfies the check.
                                passed = receipt.status in {"applied", "waiting"} and self._task_in_run(row, goal, run) and row.status == "done"
                                wake = await self._receipt_live_wait(db, receipt, goal, run) if receipt.status == "waiting" else None
                        elif check.verifier == "artifact.exists" and in_goal and arguments.project_id == goal.project_id:
                            verified_artifacts = []
                            for receipt, effect in matches:
                                if receipt.status != "applied":
                                    continue
                                for ref in receipt.target_refs:
                                    if ref["kind"] != "artifact" or (hasattr(arguments, "artifact_id") and UUID(ref["id"]) != arguments.artifact_id):
                                        continue
                                    row = await db.get(Artifact, UUID(ref["id"]))
                                    observed = receipt.observed_post_state
                                    expected = {"artifact_id": ref["id"], "artifact_type": arguments.artifact_type,
                                                "content_hash": arguments.content_hash}
                                    if (row and row.status != "deleted" and row.project_id == goal.project_id and row.artifact_type == arguments.artifact_type
                                            and row.content_hash == arguments.content_hash and isinstance(observed, dict)
                                            and all(observed.get(key) == value for key, value in expected.items())
                                            and await self._artifact_from_effect(db, row, effect, goal, run)):
                                        verified_artifacts.append(row)
                            if len(verified_artifacts) == 1:
                                passed = True
                                evidence.append(self._verification_evidence(check, "artifact", verified_artifacts[0]))
                elif check.verifier == "gate.satisfied":
                    row = await db.get(OrchestrationGate, arguments.gate_id)
                    evidence.append(self._verification_evidence(check, "gate", row, arguments.gate_id))
                    if in_goal and in_run and row and row.run_id == run.id and row.status == "accepted":
                        _selected, failure = await OrchestrationService()._gate_validation_result(db, row)
                        passed = failure is None
                elif check.verifier == "process.completed":
                    row = await db.get(OrchestrationProcessRun, arguments.process_id)
                    evidence.append(self._verification_evidence(check, "process", row, arguments.process_id))
                    if in_goal and row and row.goal_id == goal.id and row.run_id == run.id and row.status == "completed":
                        current = await OrchestrationProcessService().get_current(db, goal.id, row.process_type)
                        passed = (row.superseded_by_id is None and current is not None and current.id == row.id
                                  and (arguments.expected_fingerprint is None
                                       or (row.input_snapshot or {}).get("fingerprint") == arguments.expected_fingerprint))
                else:  # warning.resolved is goal-scoped across runs.
                    row = await db.get(OrchestrationWarning, arguments.warning_id)
                    evidence.append(self._verification_evidence(check, "warning", row, arguments.warning_id))
                    if in_goal and row and row.goal_id == goal.id:
                        await check_goal_linkage(db, goal.id, run_id=row.run_id,
                            source_process_run_id=row.source_process_run_id, related_gate_id=row.related_gate_id,
                            related_action_id=row.related_action_id, source_agent_review_id=row.source_agent_review_id,
                            related_authority_decision_id=row.related_authority_decision_id)
                        passed = not row.active and bool((row.resolved_by or "").strip()) and bool((row.resolved_reason or "").strip())
            except (ValueError, TypeError, KeyError, HTTPException):
                passed, wake = False, None
            if not evidence:
                candidates = [row for row in (await db.scalars(select(OrchestrationAction).where(
                    OrchestrationAction.action_type == "recovery_proposal"))).all()
                    if isinstance(row.request, dict) and str(row.request.get("proposal_id")) == str(proposal.proposal_id)]
                evidence.extend(self._verification_evidence(check, "action", row) for row in candidates[:32])
                if not evidence:
                    # Report the actual source row's kind; a reported blocker need not be an action.
                    for kind, model in (("warning", OrchestrationWarning), ("gate", OrchestrationGate),
                                        ("action", OrchestrationAction), ("run", OrchestrationRun),
                                        ("task", Task), ("process", OrchestrationProcessRun)):
                        row = await db.get(model, proposal.source.id)
                        if row is not None:
                            evidence.append(self._verification_evidence(check, kind, row))
                            break
                if not evidence:
                    raise ValueError("missing_verification_evidence")
            # Deduplicate checked-row evidence, and publish only bounded identities/state.
            evidence = list({(item.source_kind, item.source_id): item for item in evidence}.values())[:32]
            return VerificationResultV1(check_id=check.check_id,
                status="passed" if passed else "pending" if wake else "failed",
                code="verified" if passed else "awaiting_producer" if wake else "not_satisfied",
                evidence=evidence, wake=None if passed else wake)

    @staticmethod
    def _verification_evidence(check: VerificationCheckV1, kind: str, row: Any, identifier: UUID | None = None) -> EvidenceV1:
        identifier = row.id if row is not None else identifier
        value = {"id": str(identifier), "verifier": check.verifier, "exists": row is not None}
        if row is not None and hasattr(row, "status"):
            value["status"] = row.status
        return EvidenceV1(label=f"Checked {check.verifier} against the {kind} row.",
                          value=RecoveryConfigurationService._disclosure_value(value), source_kind=kind, source_id=identifier)

    @staticmethod
    def _verification_call(proposal: RecoveryProposalV1, call_id: str) -> RecoveryCallV1 | None:
        calls = [call for call in proposal.calls if call.call_id == call_id]
        return calls[0] if len(calls) == 1 else None

    @staticmethod
    async def _verification_origin(db: AsyncSession, proposal: RecoveryProposalV1) -> tuple | None:
        proposal_hash = _json_hash(proposal.model_dump(mode="json"))
        candidates = [row for row in (await db.scalars(select(OrchestrationAction).where(
            OrchestrationAction.action_type == "recovery_proposal"))).all()
            if isinstance(row.request, dict) and str(row.request.get("proposal_id")) == str(proposal.proposal_id)]
        if len(candidates) != 1:
            return None
        row = candidates[0]
        if _json_hash(row.request) != proposal_hash or (row.dispatch_contract or {}).get("proposal_hash") != proposal_hash:
            return None
        run = await db.get(OrchestrationRun, row.run_id)
        goal = await db.get(OrchestrationGoal, run.goal_id) if run else None
        if goal is None or run.status not in ACTIVE_RUN_STATUSES:
            return None
        return row, goal, run

    @staticmethod
    async def _verification_wrapper(db: AsyncSession, proposal: RecoveryProposalV1,
                                    call: RecoveryCallV1, run: OrchestrationRun | None) -> OrchestrationAction | None:
        query = select(OrchestrationAction).where(
            OrchestrationAction.idempotency_key == f"recovery:{proposal.proposal_id}:apply:{call.call_id}")
        if run:
            query = query.where(OrchestrationAction.run_id == run.id)
        rows = (await db.scalars(query)).all()
        if len(rows) == 1:
            return rows[0]
        # A changed key is a failed identity check, but still disclose the actual checked wrapper row.
        if not rows and run:
            candidates = (await db.scalars(select(OrchestrationAction).where(
                OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == "recovery_apply"))).all()
            rows = [row for row in candidates if isinstance(row.request, dict)
                    and row.request.get("call_id") == call.call_id
                    and row.request.get("proposal_hash") == _json_hash(proposal.model_dump(mode="json"))]
        return rows[0] if len(rows) == 1 else None

    async def _bound_receipt(self, db: AsyncSession, proposal: RecoveryProposalV1, call: RecoveryCallV1,
                             wrapper: OrchestrationAction | None, origin: tuple) -> tuple[ToolReceiptV1, OrchestrationAction] | None:
        proposal_action, goal, run = origin
        if (wrapper is None or wrapper.run_id != run.id or wrapper.action_type != "recovery_apply"
                or wrapper.status != "completed"
                or wrapper.idempotency_key != f"recovery:{proposal.proposal_id}:apply:{call.call_id}"):
            return None
        request, contract = wrapper.request or {}, wrapper.dispatch_contract or {}
        proposal_hash, request_hash = _json_hash(proposal.model_dump(mode="json")), _json_hash(call.arguments)
        if (request.get("proposal_action_id") != str(proposal_action.id) or request.get("proposal_hash") != proposal_hash
                or request.get("call_id") != call.call_id or contract.get("canonical_request_hash") != request_hash):
            return None
        try:
            receipt = ToolReceiptV1.model_validate(contract.get("tool_receipt_v1"))
            authority = await db.get(OrchestrationAuthorityDecision, UUID(str(contract.get("authority_decision_id"))))
        except (ValueError, TypeError):
            return None
        identity = receipt.execution_identity
        if (authority is None or authority.goal_id != goal.id or authority.run_id != run.id
                or authority.related_action_id != proposal_action.id or authority.status != "answered"
                or authority.selected_option != "approve_fix" or receipt.proposal_id != proposal.proposal_id
                or receipt.call_id != call.call_id or receipt.tool != f"{call.tool}@{call.tool_version}"
                or receipt.wrapper_action_id != wrapper.id or receipt.canonical_request_hash != request_hash
                or identity.proposal_id != proposal.proposal_id or identity.call_id != call.call_id
                or identity.authority_decision_id != authority.id or receipt.status not in {"applied", "waiting"}):
            return None
        effect = await db.get(OrchestrationAction, receipt.action_id)
        aggregate = call.tool == "team_configuration.apply"
        expected_key = wrapper.idempotency_key if aggregate else wrapper.idempotency_key + ":effect"
        if (effect is None or effect.run_id != run.id or effect.status != "completed"
                or effect.idempotency_key != expected_key or receipt.idempotency_key != expected_key
                or contract.get("effect_action_id") != str(effect.id)
                or contract.get("effect_idempotency_key") != expected_key
                or (aggregate and effect.id != wrapper.id)
                or (not aggregate and (effect.id == wrapper.id or effect.action_type != _ACTION_TYPES.get(call.tool)))):
            return None
        if aggregate:
            rows = [await OrchestrationProcessService().get_current(db, goal.id, kind)
                    for kind in ("agent_definition_review", "team_hierarchy")]
            if not await self._team_configuration_accepted(db, proposal, call, goal, run, rows):
                return None
        elif _json_hash(OrchestrationService().canonical_decision_request(effect.action_type, effect.request, run_id=run.id)) != request_hash:
            return None
        if not await self._receipt_targets_match(db, receipt, effect, call, goal, run):
            return None
        return receipt, effect

    @staticmethod
    def _task_in_run(task: Task | None, goal: OrchestrationGoal, run: OrchestrationRun) -> bool:
        if task is None or task.project_id != goal.project_id:
            return False
        orchestration = (task.metadata_ or {}).get("orchestration", {})
        return isinstance(orchestration, dict) and orchestration.get("run_id") == str(run.id) and orchestration.get("goal_id", str(goal.id)) == str(goal.id)

    @staticmethod
    async def _effect_task_id(db: AsyncSession, effect: OrchestrationAction) -> UUID | None:
        if effect.target_type == "task":
            return effect.target_id
        models = {"session": (Session, "task_id"), "meeting": (Meeting, "source_task_id"),
                  "graph_run": (GraphRun, "linked_task_id"), "artifact": (Artifact, "linked_task_id")}
        if effect.target_type in models and effect.target_id:
            model, attribute = models[effect.target_type]
            row = await db.get(model, effect.target_id)
            return getattr(row, attribute) if row else None
        return None

    async def _artifact_from_effect(self, db: AsyncSession, artifact: Artifact, effect: OrchestrationAction,
                                    goal: OrchestrationGoal, run: OrchestrationRun) -> bool:
        task_id = await self._effect_task_id(db, effect)
        task = await db.get(Task, artifact.linked_task_id) if artifact.linked_task_id else None
        return (artifact.project_id == goal.project_id and task_id is not None and task_id == artifact.linked_task_id
                and self._task_in_run(task, goal, run))

    async def _receipt_targets_match(self, db: AsyncSession, receipt: ToolReceiptV1, effect: OrchestrationAction,
                                     call: RecoveryCallV1, goal: OrchestrationGoal, run: OrchestrationRun) -> bool:
        refs = receipt.target_refs
        if any(set(ref) != {"kind", "id"} or not isinstance(ref["kind"], str) or not isinstance(ref["id"], str) for ref in refs):
            return False
        pairs = {(ref["kind"], UUID(ref["id"])) for ref in refs}
        if len(pairs) != len(refs) or (effect.target_type, effect.target_id) not in pairs:
            return False
        task_id = await self._effect_task_id(db, effect)
        if task_id is not None and not self._task_in_run(await db.get(Task, task_id), goal, run):
            return False
        explicit_task = call.arguments.get("task_id") or call.arguments.get("plan_task_id")
        if explicit_task is not None and task_id != UUID(str(explicit_task)):
            return False
        service = OrchestrationService()
        if call.tool == "orchestration.schedule_meeting":
            if effect.target_type != "meeting":
                return False
            expected_task_id = await service._meeting_task_id(db, run.id, call.arguments)
            _project, expected_task, _gate = await service._coordination_task_and_gate(
                db, run.id, expected_task_id,
                UUID(call.arguments["gate_id"]) if call.arguments.get("gate_id") else None,
            )
            if task_id != expected_task.id:
                return False
        elif call.tool == "orchestration.start_graph":
            if effect.target_type != "graph_run":
                return False
            _project, expected_task, _gate, artifact_id = await service._graph_subject(
                db, run.id, call.arguments["subject_type"], UUID(call.arguments["subject_id"]),
            )
            graph_run = await db.get(GraphRun, effect.target_id)
            if (graph_run is None or task_id != expected_task.id
                    or graph_run.graph_id != UUID(call.arguments["graph_id"]) or graph_run.artifact_id != artifact_id):
                return False
        sealed: set[tuple[str, UUID]] = set()
        for key, kind in (("agent_id", "agent"), ("task_id", "task"), ("plan_task_id", "task"), ("gate_id", "gate")):
            if call.arguments.get(key):
                sealed.add((kind, UUID(str(call.arguments[key]))))
        if call.tool == "team_configuration.apply":
            sealed.update(("agent", UUID(item["agent_id"])) for key in ("agents", "eligibility", "assignments") for item in call.arguments[key])
        if call.tool == "orchestration.start_graph":
            sealed.add((call.arguments["subject_type"], UUID(call.arguments["subject_id"])))
        for kind, identifier in pairs:
            if kind == "artifact":
                artifact = await db.get(Artifact, identifier)
                if artifact is None or not await self._artifact_from_effect(db, artifact, effect, goal, run):
                    return False
            elif kind == "task":
                task = await db.get(Task, identifier)
                if not self._task_in_run(task, goal, run) or (identifier != task_id and (kind, identifier) not in sealed):
                    return False
            elif kind == "agent":
                agent = await db.get(Agent, identifier)
                if agent is None or not agent.is_active or (kind, identifier) not in sealed:
                    return False
            elif kind == "gate":
                gate = await db.get(OrchestrationGate, identifier)
                if gate is None or gate.run_id != run.id or (kind, identifier) not in sealed:
                    return False
            elif (kind, identifier) != (effect.target_type, effect.target_id):
                return False
            else:
                model = {"session": Session, "meeting": Meeting, "graph_run": GraphRun}.get(kind)
                row = await db.get(model, identifier) if model else None
                if row is None or (hasattr(row, "project_id") and row.project_id != goal.project_id):
                    return False
        return True

    @staticmethod
    def _configuration_gates(definition: OrchestrationProcessRun | None, hierarchy: OrchestrationProcessRun | None) -> bool:
        if definition is None or hierarchy is None or definition.status != "completed" or hierarchy.status != "completed":
            return False
        outputs = hierarchy.outputs or {}
        gates = outputs.get("gates", {})
        required = ("team_structure_reviewed", "agent_definitions_reviewed", "missing_capabilities_handled",
                    "independent_verification_possible_or_overridden")
        return ((definition.outputs or {}).get("gates", {}).get("agent_definitions_reviewed") is True
                and all(gates.get(key) is True for key in required)
                and (gates.get("required_work_functions_mapped") is True or bool(outputs.get("documented_gaps"))))

    async def _team_configuration_accepted(self, db: AsyncSession, proposal: RecoveryProposalV1, call: RecoveryCallV1,
                                           goal: OrchestrationGoal, run: OrchestrationRun, rows: list) -> bool:
        definition, hierarchy = rows
        if (not self._configuration_gates(definition, hierarchy)
                or any(row.goal_id != goal.id or row.run_id != run.id or row.superseded_by_id is not None for row in rows)):
            return False
        provenance = {"proposal_id": str(proposal.proposal_id), "proposal_hash": _json_hash(proposal.model_dump(mode="json")),
                      "call_id": call.call_id, "canonical_request_hash": _json_hash(call.arguments)}
        if any(not isinstance((row.outputs or {}).get("recovery_provenance"), dict)
               or any(row.outputs["recovery_provenance"].get(key) != value for key, value in provenance.items()) for row in rows):
            return False
        ordered = RecoveryConfigurationService._ordered
        arguments = call.arguments
        surfaces = await self.configuration_service._accepted_surfaces(db, goal)
        expected_members = {key: arguments["hierarchy"][key] for key in ("team_leads", "contributors", "reviewers", "validators", "specialists")}
        expected_surfaces = {"eligibility": arguments["eligibility"], "assignments": arguments["assignments"],
                             "manager": arguments["hierarchy"]["manager"], "documented_gaps": arguments["documented_gaps"]}
        if (any(ordered(surfaces[key]) != ordered(value) for key, value in expected_surfaces.items())
                or ordered(surfaces["reporting_hierarchy"]["role_membership"]) != ordered(expected_members)):
            return False
        # Compare full sealed definitions, including unchanged fields, using the same safe disclosure projection.
        sealed_agents = {}
        sealed_fingerprints = {}
        for change in call.changes:
            target = sealed_agents if change.target_type == "agent" else sealed_fingerprints if change.target_type == "configuration_fingerprint" else None
            if target is None:
                if change.target_type == "reporting_hierarchy" and ordered(surfaces["reporting_hierarchy"]) != ordered(change.after):
                    return False
                continue
            if change.target_id == "aggregate":
                target.update({identifier: value["after"] for identifier, value in change.after.items()})
            else:
                target[str(change.target_id)] = change.after
        if set(sealed_agents) != {item["agent_id"] for item in arguments["agents"]}:
            return False
        for identifier, expected in sealed_agents.items():
            agent = await db.get(Agent, UUID(identifier))
            if agent is None or not agent.is_active:
                return False
            actual = RecoveryConfigurationService._disclosure_value(RecoveryConfigurationService._state_value(RecoveryConfigurationService._from_agent(agent)))
            if ordered(actual) != ordered(expected):
                return False
        covered = set((definition.outputs or {}).get("target_ids", []))
        if not {item["agent_id"] for item in arguments["eligibility"]}.issubset(covered):
            return False
        review = AgentDefinitionReviewProcess()
        definition_fp = await review.current_fingerprint(db, goal, run)
        coverage_fp = await review.current_coverage_fingerprint(db, goal, run, covered_target_ids=covered)
        hierarchy_fp = await TeamHierarchyProcess().current_fingerprint(db, goal, run)
        expected_fps = sealed_fingerprints.get(str(goal.id))
        return (expected_fps is not None and expected_fps == {"coverage": coverage_fp, "hierarchy": hierarchy_fp}
                and definition.outputs.get("fingerprint") == definition_fp
                and (definition.input_snapshot or {}).get("fingerprint") == definition_fp
                and definition.outputs.get("coverage_fingerprint") == coverage_fp
                and hierarchy.outputs.get("fingerprint") == hierarchy_fp
                and (hierarchy.input_snapshot or {}).get("fingerprint") == hierarchy_fp)

    async def _receipt_live_wait(self, db: AsyncSession, receipt: ToolReceiptV1,
                                 goal: OrchestrationGoal, run: OrchestrationRun) -> Any:
        if receipt.wait is None:
            return None
        normalized = normalize_wake_when(receipt.wait.model_dump(mode="json"))
        task_ids = {UUID(ref["id"]) for ref in receipt.target_refs if ref["kind"] == "task"}
        for task_id in task_ids:
            task = await db.get(Task, task_id)
            if not self._task_in_run(task, goal, run) or task.status not in {"ready", "in_progress", "blocked"}:
                continue
            sessions = (await db.scalars(select(Session).where(Session.task_id == task_id, Session.status.in_(("pending", "running"))))).all()
            for event in normalized["events"]:
                matcher = event["matcher"]
                if (sessions and event["event_type"] in {"session.completed", "session.failed", "session.cancelled"}
                        and matcher.get("task_id") == str(task_id)
                        and ("session_id" not in matcher or matcher["session_id"] in {str(session.id) for session in sessions})):
                    return receipt.wait
        return None
