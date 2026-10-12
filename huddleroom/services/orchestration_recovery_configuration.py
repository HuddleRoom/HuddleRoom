"""Sealed team-configuration proposals: read-only preparation and the atomic apply."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Literal

from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, ConfigDict, Field, field_validator

from huddleroom.models.agent import Agent
from huddleroom.models.orchestration import ACTIVE_RUN_STATUSES, OrchestrationAction, OrchestrationGoal, OrchestrationRun
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision, OrchestrationProcessRun
from huddleroom.schemas.agent import AgentCreate, AgentUpdate
from huddleroom.schemas.orchestration_recovery import (
    ChangeV1, RecoveryCallV1, RecoveryProposalV1, TeamConfigurationApplyV1, ToolReceiptV1,
    _AssignmentV1, _DocumentedGapV1, _EligibilityV1,
)
from huddleroom.services.agent_service import AgentService, _effective_cli_runtime, _has_cli_runtime
from huddleroom.services.orchestration_agent_definition_analyzer import redact_semantic_payload
from huddleroom.services.orchestration_agent_definition_review import (
    AgentDefinitionReviewProcess,
    recovery_eligibility_assessment,
)
from huddleroom.services.orchestration_roster_mutation import (
    AgentRosterMutationCoordinator, CanonicalAgentState, RosterMutationPlan, RosterStateChanged,
    ValidatedRosterDelta, fresh_preview, holds_goal_lock, preview_roster_delta,
)
from huddleroom.services.orchestration_team_hierarchy import (
    TeamHierarchyProcess,
    derive_recovery_hierarchy,
    recovery_manager_matches,
)

TOOL = "team_configuration.apply"
_MEMBER_KEYS = ("team_leads", "contributors", "reviewers", "validators", "specialists")


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


@contextmanager
def _fresh_reads(db: AsyncSession) -> Iterator[None]:
    """Repopulate every ORM SELECT from the database (no stale identity-map reads)."""
    def repopulate(state) -> None:
        if state.is_select:
            state.update_execution_options(populate_existing=True)

    event.listen(db.sync_session, "do_orm_execute", repopulate)
    try:
        yield
    finally:
        event.remove(db.sync_session, "do_orm_execute", repopulate)


@dataclass(frozen=True)
class _BoundApply:
    """ORM-free identity of one approved apply call (safe across the coordinator's commits)."""
    goal_id: uuid.UUID
    run_id: uuid.UUID
    proposal_action_id: uuid.UUID
    proposal_hash: str
    wrapper_id: uuid.UUID
    wrapper_key: str
    authority_id: uuid.UUID
    request_hash: str


class _CanonicalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _CanonicalAgentDefinition(AgentCreate):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _CanonicalAgentUpdate(AgentUpdate):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _CanonicalCreateAgent(_CanonicalModel):
    operation: Literal["create"]
    agent_id: uuid.UUID
    definition: _CanonicalAgentDefinition


class _CanonicalUpdateAgent(_CanonicalModel):
    operation: Literal["update"]
    agent_id: uuid.UUID
    set: _CanonicalAgentUpdate


class _CanonicalEligibility(_CanonicalModel):
    agent_id: uuid.UUID
    work_functions: list[str] = Field(min_length=1, max_length=7)

    @field_validator("work_functions")
    @classmethod
    def _functions(cls, value: list[str]) -> list[str]:
        return _EligibilityV1._functions(value)


class _CanonicalAssignment(_CanonicalModel):
    work_function: str
    agent_id: uuid.UUID

    @field_validator("work_function")
    @classmethod
    def _function(cls, value: str) -> str:
        return _AssignmentV1._function(value)


class _CanonicalAgentManager(_CanonicalModel):
    kind: Literal["agent"]
    agent_id: uuid.UUID


class _CanonicalHumanManager(_CanonicalModel):
    kind: Literal["human"]
    user_id: uuid.UUID


class _CanonicalNoManager(_CanonicalModel):
    kind: Literal["none"]


class _CanonicalHierarchy(_CanonicalModel):
    manager: _CanonicalAgentManager | _CanonicalHumanManager | _CanonicalNoManager
    team_leads: list[uuid.UUID]
    contributors: list[uuid.UUID]
    reviewers: list[uuid.UUID]
    validators: list[uuid.UUID]
    specialists: list[uuid.UUID]


class _CanonicalGap(_DocumentedGapV1):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CanonicalTeamConfigurationApplyV1(_CanonicalModel):
    agents: list[_CanonicalCreateAgent | _CanonicalUpdateAgent]
    eligibility: list[_CanonicalEligibility]
    assignments: list[_CanonicalAssignment]
    hierarchy: _CanonicalHierarchy
    documented_gaps: list[_CanonicalGap]


@dataclass(frozen=True)
class CanonicalTeamConfigurationChange:
    arguments: dict[str, Any]
    changes: tuple[ChangeV1, ...]
    roster_delta: ValidatedRosterDelta
    roster_plan: RosterMutationPlan


class RecoveryConfigurationService:
    async def prepare(self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun,
                      draft: TeamConfigurationApplyV1) -> CanonicalTeamConfigurationChange:
        # Reject at sealing what apply can never accept, so no doomed approve_fix card exists.
        if goal.weight == "trivial":
            raise ValueError("unsupported_recovery_goal")
        with db.no_autoflush:
            current = {agent.id: agent for agent in (await db.execute(select(Agent))).scalars()}
        refs: dict[str, uuid.UUID] = {}
        update_ids: set[uuid.UUID] = set()
        creates: list[CanonicalAgentState] = []
        updates: list[CanonicalAgentState] = []
        agent_changes: list[tuple[uuid.UUID, dict[str, Any], dict[str, Any]]] = []
        for item in draft.agents:
            if item.operation == "create":
                if item.ref in refs:
                    raise ValueError("duplicate_agent_ref")
                identifier = uuid.uuid4(); refs[item.ref] = identifier; definition = item.definition
                # Reuse the public input contract and its pure CLI-runtime invariant.
                create = AgentCreate.model_validate(definition.model_dump())
                if create.adapter_type == "cli" and not _has_cli_runtime(
                    _effective_cli_runtime(create.cli_runtime, create.config)
                ):
                    raise ValueError("cli_runtime_required")
                state = CanonicalAgentState(identifier, definition.name, definition.role, definition.description, definition.provider,
                    definition.model, definition.system_prompt, definition.adapter_type, definition.cli_runtime,
                    tuple(definition.capabilities), copy.deepcopy(definition.config), True)
                creates.append(state)
                agent_changes.append((identifier, {}, self._state_value(state)))
            else:
                if item.agent_id in update_ids:
                    raise ValueError("duplicate_agent_update")
                update_ids.add(item.agent_id)
                agent = current.get(item.agent_id)
                if agent is None or not agent.is_active: raise ValueError("out_of_scope_agent")
                update, state = self._normalize_update(agent, item.set)
                updates.append(state)
                agent_changes.append((agent.id, self._state_value(self._from_agent(agent)), self._state_value(state)))
        delta = ValidatedRosterDelta(tuple(creates), tuple(updates))
        if not creates and not any(self._from_agent(current[item.agent_id]) != state for item, state in zip((x for x in draft.agents if x.operation == "update"), updates)):
            raise ValueError("no_op_roster_delta")
        plan = await preview_roster_delta(db, delta, origin_goal_id=goal.id)
        arguments = self._canonical_arguments(draft, refs)
        self._validate_references(arguments, {agent.id for agent in current.values() if agent.is_active} | set(refs.values()))
        canonical = CanonicalTeamConfigurationApplyV1.model_validate(arguments)
        # Seal-time: the full post-state apply would validate (manager, eligibility, assignments, hierarchy).
        self._validate_at_seal(goal, canonical, plan)
        arguments = canonical.model_dump(mode="json", exclude_unset=True)
        before = await self._accepted_surfaces(db, goal)
        changes = self._agent_changes(agent_changes)
        changes.extend(self._aggregate_changes(goal.id, before, arguments))
        changes.extend(self._fingerprint_changes(plan))
        if len(changes) > 32:
            raise ValueError("change_set_too_large")
        return CanonicalTeamConfigurationChange(arguments, tuple(changes), delta, plan)

    # ---- apply ------------------------------------------------------------------------------

    async def apply(
        self, db: AsyncSession, proposal: RecoveryProposalV1, call: RecoveryCallV1,
        wrapper: OrchestrationAction, authority_decision: OrchestrationAuthorityDecision,
    ) -> ToolReceiptV1:
        """Atomically apply one approved ``team_configuration.apply@1`` call.

        Roster-first (coordinator): agent rows, both recovery-sourced accepted
        snapshots, collateral invalidations, the receipt and the wrapper completion
        commit in one transaction, or nothing does. The caller must hold no goal lock
        and passes an AUTOBEGIN session; this method owns the commit. A stale sealed
        plan raises ``RosterStateChanged`` (never re-planned); any other failure raises
        and leaves the wrapper reserved.
        """
        existing = await self.reconcile(db, proposal, call, wrapper)
        if existing is not None:
            return existing
        if holds_goal_lock():
            raise RuntimeError("team configuration must be applied roster-first, without a goal lock")
        wrapper_id, authority_id = wrapper.id, authority_decision.id
        with _fresh_reads(db):
            bound = await self._bind(db, proposal, call, wrapper_id, authority_id)
            goal = await db.get(OrchestrationGoal, bound.goal_id)
            canonical, delta, plan = await self._reconstruct(db, goal, call)
            self._validate_post_state(goal, canonical, plan)

        async def mutation(session: AsyncSession, locked_delta: ValidatedRosterDelta, locked_plan: RosterMutationPlan):
            from huddleroom.services.orchestration_recovery_registry import RecoveryToolRegistry

            with _fresh_reads(session):
                rebound = await self._bind(session, proposal, call, wrapper_id, authority_id)
                if rebound != bound:
                    raise RosterStateChanged("recovery approval changed")
                locked_goal = await session.get(OrchestrationGoal, bound.goal_id, populate_existing=True)
                locked_run = await session.get(OrchestrationRun, bound.run_id, populate_existing=True)
                resources = await RecoveryToolRegistry(self).capture_resources(session, locked_goal, locked_run, proposal)
            if resources.fingerprint != proposal.resource_fingerprint:
                raise RosterStateChanged("recovery resources changed since approval")
            await AgentService()._apply_delta_rows(session, locked_delta)  # pylint: disable=protected-access
            receipt = await self._persist_acceptance(session, proposal, call, bound, canonical, locked_goal, locked_run, locked_plan)
            return receipt

        result = await AgentRosterMutationCoordinator().apply(
            db, delta, plan, mutation, origin_goal_id=bound.goal_id, origin_run_id=bound.run_id,
            provenance={"proposal_action_id": bound.proposal_action_id, "proposal_id": proposal.proposal_id,
                        "authority_decision_id": bound.authority_id},
        )
        return result.value

    async def reconcile(
        self, db: AsyncSession, proposal: RecoveryProposalV1, call: RecoveryCallV1,
        wrapper: OrchestrationAction,
    ) -> ToolReceiptV1 | None:
        """The committed receipt of this exact apply, or None when it has not been applied.

        Snapshots, receipt and wrapper completion commit together, so a completed wrapper
        proves the snapshots exist; replay verifies they name this proposal/hash.
        """
        row = await db.get(OrchestrationAction, wrapper.id, populate_existing=True)
        contract = (row.dispatch_contract or {}) if row is not None else {}
        if row is None or row.status != "completed" or "tool_receipt_v1" not in contract:
            return None
        receipt = ToolReceiptV1.model_validate(contract["tool_receipt_v1"])
        key = f"recovery:{proposal.proposal_id}:apply:{call.call_id}"
        proposal_hash = _json_hash(proposal.model_dump(mode="json"))
        run = await db.get(OrchestrationRun, row.run_id)
        snapshots = list((await db.scalars(select(OrchestrationProcessRun).where(
            OrchestrationProcessRun.goal_id == run.goal_id,
            OrchestrationProcessRun.trigger_reason == "approved_recovery",
        ))).all()) if run is not None else []
        named = {item.process_type for item in snapshots
                 if ((item.outputs or {}).get("recovery_provenance") or {}).get("proposal_id") == str(proposal.proposal_id)
                 and item.outputs["recovery_provenance"].get("proposal_hash") == proposal_hash}
        if (row.action_type != "recovery_apply" or row.idempotency_key != key or receipt.idempotency_key != key
                or receipt.proposal_id != proposal.proposal_id or receipt.call_id != call.call_id
                or receipt.tool != f"{TOOL}@1" or receipt.wrapper_action_id != row.id or receipt.action_id != row.id
                or receipt.canonical_request_hash != _json_hash(call.arguments)
                or named != {"agent_definition_review", "team_hierarchy"}):
            raise ValueError("recovery_receipt_mismatch")
        return receipt

    async def _bind(self, db: AsyncSession, proposal: RecoveryProposalV1, call: RecoveryCallV1,
                    wrapper_id: uuid.UUID, authority_id: uuid.UUID) -> _BoundApply:
        """Prove the call, wrapper and authority belong to this exact approved proposal."""
        sealed = [item for item in proposal.calls if item.call_id == call.call_id]
        if len(sealed) != 1 or sealed[0] != call or call.tool != TOOL or call.tool_version != "1":
            raise ValueError("call_not_in_approved_proposal")
        proposal_hash = _json_hash(proposal.model_dump(mode="json"))
        actions = (await db.scalars(select(OrchestrationAction).where(
            OrchestrationAction.idempotency_key == f"recovery:{proposal.proposal_id}:proposal",
            OrchestrationAction.action_type == "recovery_proposal",
        ))).all()
        action = actions[0] if len(actions) == 1 else None
        if (action is None or action.status != "reserved" or _json_hash(action.request) != proposal_hash
                or (action.dispatch_contract or {}).get("proposal_hash") != proposal_hash):
            raise ValueError("proposal_not_approved")
        key, request_hash = f"recovery:{proposal.proposal_id}:apply:{call.call_id}", _json_hash(call.arguments)
        wrapper = await db.get(OrchestrationAction, wrapper_id, populate_existing=True)
        request, contract = ((wrapper.request or {}), (wrapper.dispatch_contract or {})) if wrapper else ({}, {})
        if (wrapper is None or wrapper.action_type != "recovery_apply" or wrapper.status != "reserved"
                or wrapper.idempotency_key != key or wrapper.run_id != action.run_id
                or request.get("proposal_action_id") != str(action.id) or request.get("proposal_hash") != proposal_hash
                or request.get("call_id") != call.call_id or contract.get("authority_decision_id") != str(authority_id)
                or contract.get("canonical_request_hash") != request_hash or "tool_receipt_v1" in contract):
            raise ValueError("wrapper_not_bound")
        authority = await db.get(OrchestrationAuthorityDecision, authority_id, populate_existing=True)
        run = await db.get(OrchestrationRun, action.run_id, populate_existing=True)
        if (authority is None or run is None or authority.related_action_id != action.id
                or authority.decision_key != f"recovery:{proposal.proposal_id}:{proposal_hash}"
                or authority.status != "answered" or authority.selected_option != "approve_fix"
                or authority.run_id != run.id or authority.goal_id != run.goal_id
                or run.status not in ACTIVE_RUN_STATUSES):
            raise ValueError("authority_not_bound")
        return _BoundApply(run.goal_id, run.id, action.id, proposal_hash, wrapper.id, key, authority.id, request_hash)

    async def _reconstruct(self, db: AsyncSession, goal: OrchestrationGoal, call: RecoveryCallV1):
        """Rebuild the sealed change from ``call.arguments`` and require it to equal the approval.

        Never canonicalizes a new delta: any drift in the roster, accepted surfaces or
        affected-goal fingerprints makes the reconstruction differ and the apply stale.
        """
        canonical = CanonicalTeamConfigurationApplyV1.model_validate(call.arguments)
        current = {agent.id: agent for agent in (await db.scalars(select(Agent))).all()}
        creates, updates, agent_changes = [], [], []
        for item in canonical.agents:
            if item.operation == "create":
                definition = AgentCreate.model_validate(item.definition.model_dump())
                AgentService.validate_create(definition)
                if item.agent_id in current:
                    raise RosterStateChanged("recovery agent already exists")
                state = CanonicalAgentState(item.agent_id, definition.name, definition.role, definition.description,
                    definition.provider, definition.model, definition.system_prompt, definition.adapter_type,
                    definition.cli_runtime, tuple(definition.capabilities), copy.deepcopy(definition.config), True)
                creates.append(state)
                agent_changes.append((state.id, {}, self._state_value(state)))
            else:
                agent = current.get(item.agent_id)
                if agent is None or not agent.is_active:
                    raise RosterStateChanged("recovery agent is no longer active")
                _update, state = self._normalize_update(agent, item.set)
                updates.append(state)
                agent_changes.append((agent.id, self._state_value(self._from_agent(agent)), self._state_value(state)))
        delta = ValidatedRosterDelta(tuple(creates), tuple(updates))
        try:
            plan = await fresh_preview(db, delta, origin_goal_id=goal.id)
        except ValueError as exc:
            raise RosterStateChanged(str(exc)) from exc
        changes = self._agent_changes(agent_changes)
        changes.extend(self._aggregate_changes(goal.id, await self._accepted_surfaces(db, goal), call.arguments))
        changes.extend(self._fingerprint_changes(plan))
        if [item.model_dump(mode="json") for item in changes] != [item.model_dump(mode="json") for item in call.changes]:
            raise RosterStateChanged("team configuration changed since approval")
        return canonical, delta, plan

    def _validate_at_seal(self, goal: OrchestrationGoal, canonical: "CanonicalTeamConfigurationApplyV1",
                          plan: RosterMutationPlan) -> None:
        """Seal-time call site of the full post-state check; apply calls ``_validate_post_state`` directly."""
        self._validate_post_state(goal, canonical, plan)

    @staticmethod
    def _validate_post_state(goal: OrchestrationGoal, canonical: "CanonicalTeamConfigurationApplyV1",
                             plan: RosterMutationPlan) -> None:
        """Validate the complete proposed post-state before any write."""
        if goal.weight == "trivial":
            raise ValueError("unsupported_recovery_goal")
        after = {agent.id: agent for agent in plan.after_roster.agents}  # active agents only
        eligibility: dict[str, list[str]] = {}
        for item in canonical.eligibility:
            if item.agent_id not in after:
                raise ValueError("inactive_or_unknown_agent")
            if str(item.agent_id) in eligibility:
                raise ValueError("duplicate_eligibility")
            recovery_eligibility_assessment(after[item.agent_id], item.work_functions)
            eligibility[str(item.agent_id)] = list(item.work_functions)
        role_to_agent: dict[str, str] = {}
        for item in canonical.assignments:
            if item.agent_id not in after:
                raise ValueError("inactive_or_unknown_agent")
            if item.work_function in role_to_agent:
                raise ValueError("duplicate_function_assignment")
            role_to_agent[item.work_function] = str(item.agent_id)
        hierarchy = canonical.hierarchy
        members = {key: [str(value) for value in getattr(hierarchy, key)] for key in _MEMBER_KEYS}
        if any(uuid.UUID(value) not in after for values in members.values() for value in values):
            raise ValueError("inactive_or_unknown_agent")
        if not recovery_manager_matches(goal, hierarchy.manager.model_dump(mode="json")):
            raise ValueError("manager_mismatch")
        derive_recovery_hierarchy(goal, role_to_agent, eligibility, members,
                                  [gap.model_dump(mode="json") for gap in canonical.documented_gaps])

    async def _persist_acceptance(self, db: AsyncSession, proposal: RecoveryProposalV1, call: RecoveryCallV1,
                                  bound: _BoundApply, canonical: "CanonicalTeamConfigurationApplyV1",
                                  goal: OrchestrationGoal, run: OrchestrationRun,
                                  plan: RosterMutationPlan) -> ToolReceiptV1:
        """Snapshots, receipt and wrapper completion (inside the coordinator's transaction)."""
        from huddleroom.services.orchestration_service import OrchestrationService

        common = {"proposal_id": proposal.proposal_id, "proposal_hash": bound.proposal_hash,
                  "authority_decision_id": bound.authority_id, "evidence": tuple(proposal.evidence)}
        payload = {"call_id": call.call_id, "canonical_request_hash": bound.request_hash}
        agent_ids = list(dict.fromkeys(item.agent_id for item in canonical.agents))
        definition = await AgentDefinitionReviewProcess().persist_recovery_acceptance(
            db, goal, run, accepted_payload={
                **payload, "changed_agent_ids": [str(item) for item in agent_ids],
                "eligibility": {str(item.agent_id): list(item.work_functions) for item in canonical.eligibility},
            }, **common,
        )
        hierarchy_arguments = call.arguments["hierarchy"]
        hierarchy = await TeamHierarchyProcess().persist_recovery_acceptance(
            db, goal, run, accepted_payload={
                **payload, "assignments": call.arguments["assignments"],
                "manager": hierarchy_arguments["manager"],
                **{key: hierarchy_arguments[key] for key in _MEMBER_KEYS},
                "documented_gaps": call.arguments["documented_gaps"],
            }, **common,
        )
        expected = next(item for item in plan.goal_fingerprints if item.goal_id == goal.id)
        if ((definition.outputs or {}).get("coverage_fingerprint"), (hierarchy.outputs or {}).get("fingerprint")) != (
            expected.after_coverage_fingerprint, expected.after_hierarchy_fingerprint,
        ):
            raise ValueError("post_state_differs_from_approved_fingerprints")
        receipt = ToolReceiptV1.model_validate({
            "proposal_id": str(proposal.proposal_id), "call_id": call.call_id, "tool": f"{TOOL}@1",
            "wrapper_action_id": str(bound.wrapper_id), "action_id": str(bound.wrapper_id),
            "idempotency_key": bound.wrapper_key, "canonical_request_hash": bound.request_hash,
            "execution_identity": {"proposal_id": str(proposal.proposal_id), "call_id": call.call_id,
                                   "authority_decision_id": str(bound.authority_id), "purpose": "apply"},
            "status": "applied",
            "target_refs": [{"kind": "agent", "id": str(item)} for item in agent_ids],
            "observed_post_state": {
                "agent_ids": [str(item) for item in agent_ids],
                "definition_process_id": str(definition.id), "hierarchy_process_id": str(hierarchy.id),
                "affected_goal_ids": [str(item) for item in plan.affected_goal_ids],
            },
            "evidence": [
                {"label": f"Recovery-sourced {row.process_type} snapshot accepted.",
                 "value": {"process_type": row.process_type, "status": row.status},
                 "source_kind": "process", "source_id": str(row.id)}
                for row in (definition, hierarchy)
            ],
        })
        wrapper = await db.get(OrchestrationAction, bound.wrapper_id)
        wrapper.dispatch_contract = {
            **(wrapper.dispatch_contract or {}), "tool_receipt_v1": receipt.model_dump(mode="json"),
            "effect_action_id": str(wrapper.id), "effect_idempotency_key": bound.wrapper_key,
        }
        # The wrapper row has one target column: the first changed agent (prepare rejects an
        # empty delta); the full changed-agent list lives in receipt.target_refs.
        await OrchestrationService()._mark_action_completed(db, wrapper, "agent", agent_ids[0])  # pylint: disable=protected-access
        return receipt

    @staticmethod
    def _from_agent(agent: Agent) -> CanonicalAgentState:
        return CanonicalAgentState(agent.id, agent.name, agent.role, agent.description, agent.provider, agent.model,
            agent.system_prompt, agent.adapter_type, agent.cli_runtime, tuple(agent.capabilities or ()),
            copy.deepcopy(agent.config or {}), bool(agent.is_active))

    @staticmethod
    def _disclosure_value(value: Any) -> Any:
        """Remove forbidden credential keys after semantic value redaction."""
        def project(item: Any) -> Any:
            if isinstance(item, list):
                return [project(nested) for nested in item]
            if not isinstance(item, dict):
                return item
            blocked = {"apikey", "accesstoken", "authorization", "password", "token", "secret", "bearer"}
            clean = {}
            for key, nested in item.items():
                if re.sub(r"[^a-z0-9]", "", key.lower()) in blocked:
                    clean["[REDACTED]"] = "[REDACTED]"
                else:
                    clean[key] = project(nested)
            return clean

        return project(redact_semantic_payload(value))

    @classmethod
    def _state_value(cls, state: CanonicalAgentState) -> dict[str, Any]:
        return cls._disclosure_value({"name": state.name, "role": state.role, "description": state.description,
            "provider": state.provider, "model": state.model, "system_prompt": state.system_prompt,
            "adapter_type": state.adapter_type, "cli_runtime": state.cli_runtime,
            "capabilities": list(state.capabilities), "config": copy.deepcopy(state.config),
            "is_active": state.is_active})

    @classmethod
    def _normalize_update(cls, agent: Agent, patch: Any) -> tuple[AgentUpdate, CanonicalAgentState]:
        """Mirror AgentService.update without mutating its persisted agent."""
        update = AgentUpdate.model_validate({key: copy.deepcopy(getattr(patch, key)) for key in patch.model_fields_set})
        values = {field: copy.deepcopy(getattr(agent, field)) for field in (
            "name", "role", "description", "provider", "model", "system_prompt", "adapter_type",
            "cli_runtime", "capabilities", "config", "is_active",
        )}
        scalar_fields = {"name", "role", "description", "provider", "model", "system_prompt", "adapter_type", "cli_runtime"}
        for field in update.model_fields_set & scalar_fields:
            values[field] = copy.deepcopy(getattr(update, field))
        for field in update.model_fields_set & {"capabilities", "config"}:
            value = getattr(update, field)
            if value is not None:
                values[field] = copy.deepcopy(value)
        if update.is_active is not None:
            values["is_active"] = update.is_active
        if update.adapter_type and update.adapter_type != "cli":
            values["cli_runtime"] = None
            values["config"] = {key: value for key, value in values["config"].items() if key != "cli_runtime"}
        resulting_adapter_type = update.adapter_type if "adapter_type" in update.model_fields_set else agent.adapter_type
        runtime_is_supplied = "cli_runtime" in update.model_fields_set
        config_is_supplied = "config" in update.model_fields_set and update.config is not None
        if resulting_adapter_type == "cli" and (
            (update.adapter_type == "cli" and agent.adapter_type != "cli") or runtime_is_supplied or config_is_supplied
        ) and not _has_cli_runtime(_effective_cli_runtime(
            update.cli_runtime if runtime_is_supplied else agent.cli_runtime,
            update.config if config_is_supplied else agent.config,
        )):
            raise ValueError("cli_runtime_required")
        return update, CanonicalAgentState(
            agent.id, values["name"], values["role"], values["description"], values["provider"], values["model"],
            values["system_prompt"], values["adapter_type"], values["cli_runtime"], tuple(values["capabilities"] or ()),
            values["config"] or {}, bool(values["is_active"]),
        )

    @staticmethod
    def _ordered(value: Any) -> Any:
        if isinstance(value, uuid.UUID):
            return str(value)
        if isinstance(value, list):
            return sorted((RecoveryConfigurationService._ordered(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))
        if isinstance(value, dict):
            return {key: RecoveryConfigurationService._ordered(value[key]) for key in sorted(value)}
        return value

    @classmethod
    def _canonical_arguments(cls, draft: TeamConfigurationApplyV1, refs: dict[str, uuid.UUID]) -> dict[str, Any]:
        def resolve(ref: str) -> uuid.UUID:
            return refs[ref] if ref in refs else uuid.UUID(ref)
        hierarchy = draft.hierarchy
        return {
            "agents": [
                ({"operation": "create", "agent_id": refs[item.ref], "definition": item.definition.model_dump(mode="json")}
                 if item.operation == "create" else {"operation": "update", "agent_id": item.agent_id,
                 "set": AgentUpdate.model_validate({key: getattr(item.set, key) for key in item.set.model_fields_set}).model_dump(mode="json", exclude_unset=True)})
                for item in draft.agents],
            "eligibility": [{"agent_id": resolve(item.agent_ref), "work_functions": item.work_functions} for item in draft.eligibility],
            "assignments": [{"agent_id": resolve(item.agent_ref), "work_function": item.work_function} for item in draft.assignments],
            "hierarchy": {"manager": ({"kind": "agent", "agent_id": resolve(hierarchy.manager.agent_ref)} if hierarchy.manager.kind == "agent" else hierarchy.manager.model_dump(mode="json")),
                          **{key: [resolve(ref) for ref in getattr(hierarchy, key)] for key in ("team_leads", "contributors", "reviewers", "validators", "specialists")}},
            "documented_gaps": [item.model_dump(mode="json") for item in draft.documented_gaps],
        }

    @staticmethod
    def _chunked_changes(target_type: str, scope: str, entries: list[tuple[uuid.UUID, Any, Any]]) -> list[ChangeV1]:
        changes: list[ChangeV1] = []; chunk: dict[str, Any] = {}
        for identifier, before, after in entries:
            candidate = {**chunk, str(identifier): {"before": before, "after": after}}
            try:
                ChangeV1(operation="replace", scope=scope, target_type=target_type, target_id="aggregate", after=candidate)
            except ValueError:
                if not chunk: raise ValueError("change_set_too_large")
                changes.append(ChangeV1(operation="replace", scope=scope, target_type=target_type, target_id="aggregate", after=chunk))
                chunk = {str(identifier): {"before": before, "after": after}}
                try:
                    ChangeV1(operation="replace", scope=scope, target_type=target_type, target_id="aggregate", after=chunk)
                except ValueError:
                    raise ValueError("change_set_too_large") from None
            else: chunk = candidate
        if chunk: changes.append(ChangeV1(operation="replace", scope=scope, target_type=target_type, target_id="aggregate", after=chunk))
        return changes

    @classmethod
    def _agent_changes(cls, entries: list[tuple[uuid.UUID, dict[str, Any], dict[str, Any]]]) -> list[ChangeV1]:
        if len(entries) == 1:
            identifier, before, after = entries[0]
            return [ChangeV1(operation="create" if not before else "update", scope="global", target_type="agent", target_id=identifier, before=before or None, after=after)]
        return cls._chunked_changes("agent", "global", sorted(entries, key=lambda item: str(item[0])))

    @classmethod
    def _aggregate_changes(cls, goal_id: uuid.UUID, before: dict[str, Any], arguments: dict[str, Any]) -> list[ChangeV1]:
        hierarchy = arguments["hierarchy"]
        after = {"eligibility": arguments["eligibility"], "assignments": arguments["assignments"], "manager": hierarchy["manager"],
                 "reporting_hierarchy": {
                     "role_membership": {key: hierarchy[key] for key in ("team_leads", "contributors", "reviewers", "validators", "specialists")},
                     # The draft changes membership; it has no explicit reporting-edge field.
                     "reporting_lines": before["reporting_hierarchy"]["reporting_lines"],
                 }, "documented_gaps": arguments["documented_gaps"]}
        return [ChangeV1(operation="replace", scope="goal", target_type=kind, target_id=goal_id,
                         before=cls._disclosure_value(cls._ordered(before[kind])), after=cls._disclosure_value(cls._ordered(value)))
                for kind, value in after.items() if cls._ordered(before[kind]) != cls._ordered(value)]

    @staticmethod
    def _fingerprint_changes(plan: RosterMutationPlan) -> list[ChangeV1]:
        entries = [(item.goal_id, {"coverage": item.before_coverage_fingerprint, "hierarchy": item.before_hierarchy_fingerprint}, {"coverage": item.after_coverage_fingerprint, "hierarchy": item.after_hierarchy_fingerprint}) for item in plan.goal_fingerprints if item.goal_id in plan.affected_goal_ids]
        if len(entries) == 1:
            identifier, before, after = entries[0]
            return [ChangeV1(operation="replace", scope="goal", target_type="configuration_fingerprint", target_id=identifier, before=before, after=after)]
        return RecoveryConfigurationService._chunked_changes("configuration_fingerprint", "goal", entries)

    async def _accepted_surfaces(self, db: AsyncSession, goal: OrchestrationGoal) -> dict[str, Any]:
        rows = list((await db.execute(select(OrchestrationProcessRun).where(OrchestrationProcessRun.goal_id == goal.id, OrchestrationProcessRun.superseded_by_id.is_(None), OrchestrationProcessRun.status == "completed"))).scalars())
        outputs = {row.process_type: row.outputs or {} for row in rows}; definition = outputs.get("agent_definition_review", {}); hierarchy = outputs.get("team_hierarchy", {})
        assignments = [{"agent_id": item.get("agent_id", item.get("agent_ref")), "work_function": item["work_function"]}
                       for item in hierarchy.get("assignments", [])]
        if "assignments" not in hierarchy:
            assignments = [{"agent_id": identifier, "work_function": function}
                           for function, identifier in hierarchy.get("role_to_agent", {}).items()]
        members = hierarchy.get("hierarchy") or TeamHierarchyProcess._hierarchy(
            goal, {item["work_function"]: item["agent_id"] for item in assignments},
        )
        return {"eligibility": [{"agent_id": identifier, "work_functions": functions} for identifier, functions in definition.get("eligibility", {}).items()],
                "assignments": assignments,
                "manager": ({"kind": "agent", "agent_id": goal.manager_agent_id} if goal.manager_agent_id else {"kind": "human", "user_id": goal.manager_user_id} if goal.manager_user_id else {"kind": "none"}),
                "reporting_hierarchy": {
                    "role_membership": {key: members.get(key, []) for key in ("team_leads", "contributors", "reviewers", "validators", "specialists")},
                    "reporting_lines": hierarchy.get("reporting_lines", []),
                }, "documented_gaps": hierarchy.get("documented_gaps", [])}

    @staticmethod
    def _validate_references(arguments: dict[str, Any], active_ids: set[uuid.UUID]) -> None:
        def check(ref: uuid.UUID) -> None:
            if ref not in active_ids: raise ValueError("out_of_scope_agent")
        eligibility = arguments.get("eligibility", [])
        assignments = arguments.get("assignments", [])
        for item in eligibility: check(item["agent_id"])
        for item in assignments: check(item["agent_id"])
        if len({json.dumps(item, sort_keys=True, default=str) for item in eligibility}) != len(eligibility): raise ValueError("duplicate_eligibility")
        if len({json.dumps(item, sort_keys=True, default=str) for item in assignments}) != len(assignments): raise ValueError("duplicate_assignment")
        hierarchy = arguments.get("hierarchy", {})
        manager = hierarchy.get("manager", {})
        if manager.get("kind") == "agent": check(manager["agent_id"])
        for key in ("team_leads", "contributors", "reviewers", "validators", "specialists"):
            if len(set(hierarchy.get(key, []))) != len(hierarchy.get(key, [])): raise ValueError("duplicate_hierarchy_member")
            for ref in hierarchy.get(key, []): check(ref)
