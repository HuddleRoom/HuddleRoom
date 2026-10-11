"""Read-only preparation of a sealed team-configuration proposal."""
from __future__ import annotations

import copy
import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, ConfigDict, Field, field_validator

from huddleroom.models.agent import Agent
from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun
from huddleroom.models.orchestration_process import OrchestrationProcessRun
from huddleroom.schemas.agent import AgentCreate, AgentUpdate
from huddleroom.schemas.orchestration_recovery import (
    ChangeV1, TeamConfigurationApplyV1, _AssignmentV1, _DocumentedGapV1, _EligibilityV1,
)
from huddleroom.services.agent_service import _effective_cli_runtime, _has_cli_runtime
from huddleroom.services.orchestration_agent_definition_analyzer import redact_semantic_payload
from huddleroom.services.orchestration_roster_mutation import CanonicalAgentState, ValidatedRosterDelta, RosterMutationPlan, preview_roster_delta
from huddleroom.services.orchestration_team_hierarchy import TeamHierarchyProcess


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
        arguments = CanonicalTeamConfigurationApplyV1.model_validate(arguments).model_dump(mode="json", exclude_unset=True)
        before = await self._accepted_surfaces(db, goal)
        changes = self._agent_changes(agent_changes)
        changes.extend(self._aggregate_changes(goal.id, before, arguments))
        changes.extend(self._fingerprint_changes(plan))
        if len(changes) > 32:
            raise ValueError("change_set_too_large")
        return CanonicalTeamConfigurationChange(arguments, tuple(changes), delta, plan)

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
