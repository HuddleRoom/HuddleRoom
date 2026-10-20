"""Work-function inference and LLM-orchestrated team hierarchy drafting."""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.config import settings
from huddleroom.models.agent import Agent
from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun
from huddleroom.models.orchestration_process import (
    OrchestrationAuthorityDecision,
    OrchestrationProcessRun,
)
from huddleroom.models.user import User
from huddleroom.schemas.agent import AgentCreate
from huddleroom.schemas.orchestration_recovery import EvidenceV1, _AssignmentV1
from huddleroom.services.agent_service import AgentService
from huddleroom.services.orchestration_agent_review_service import OrchestrationAgentReviewService
from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
from huddleroom.services.orchestration_goal_definition import requires_independent_verification
from huddleroom.services.orchestration_memory_service import OrchestrationMemoryService
from huddleroom.services.orchestration_process_service import (
    OrchestrationProcessService,
    effective_input_fingerprint,
    capture_input_fingerprint,
    set_effective_input_fingerprint,
    request_process_rerun,
    pending_rerun_intent,
)
from huddleroom.services.orchestration_roster_mapper import (
    PROFILES,
    WEAK_FIT_THRESHOLD,
    OrchestrationRosterMapper,
)
from huddleroom.services.orchestration_warning_service import OrchestrationWarningService
from huddleroom.services.orchestration_team_hierarchy_analyzer import TeamHierarchyAnalyzer, parse_team_hierarchy_analysis
from huddleroom.services.orchestration_agent_definition_analyzer import redact_semantic_payload
from huddleroom.services.orchestration_llm_decision_adapter import _safe_completion_error


PROCESS_TYPE = "team_hierarchy"
PROCESS_VERSION = 2
# Legacy names retained for imported fixtures; new writes use explicit rerun intent.
PROPOSAL_RESOLUTION_RERUN_SENTINEL = "rerun after proposal resolution"
RESOLVED_VALIDATION_FAILED_RERUN_SENTINEL = "rerun after resolved hierarchy validation failed"
HIERARCHY_CHANGES_REQUESTED_RERUN_SENTINEL = "rerun after hierarchy changes requested"
APPROVAL_DECISION_KEY = "team_hierarchy:approval"
MEMORY_SECTION_KEY = "team_hierarchy"
MEMORY_TOC_ORDER = 40
TEAM_HIERARCHY_SKIP_MESSAGE = (
    "Team hierarchy review was skipped. Work may be delegated without clear "
    "ownership or escalation paths."
)
NO_INDEPENDENT_VERIFIER_MESSAGE = (
    "No safe independent verifier was found. Completion may require human "
    "override for independent verification gates."
)
WORK_FUNCTION_ORDER = (
    "planning",
    "investigation",
    "implementation",
    "review",
    "validation",
    "summarization",
)
_SPECIALIST_FUNCTIONS = ("investigation", "implementation", "review", "validation")
_VERIFIER_FUNCTIONS = {"review", "validation"}
_ARTIFACT_VERB_RE = re.compile(
    r"\b(?:write|writes|writing|wrote|written|save|saves|saving|saved|generate|generates|generating|generated|drafting|drafted)\b"
)
_STANDARD_TIER_BLOCKING_GAPS = frozenset({
    "no_independent_verifier",
    "agent_definition_review_skipped",
})


def semantic_fingerprint_from_projection(payload: dict) -> str:
    """Hash a pre-built hierarchy projection without database access."""
    return hashlib.sha256(json.dumps(
        redact_semantic_payload(payload), sort_keys=True, separators=(",", ":"), default=str
    ).encode()).hexdigest()


def manager_from_projection(goal: OrchestrationGoal, agents: list[Agent] = ()) -> dict[str, str | None]:
    if goal.manager_agent_id is not None:
        manager: dict[str, str | None] = {"kind": "agent", "id": str(goal.manager_agent_id)}
        selected = next((agent for agent in agents if agent.id == goal.manager_agent_id), None)
        return {
            **manager,
            "name": selected.name if selected else None,
            "role": selected.role if selected else None,
        }
    if goal.manager_user_id is not None:
        return {"kind": "human", "id": str(goal.manager_user_id)}
    return {"kind": "none", "id": None}


def hierarchy_analysis_input_from_projection(
    goal: OrchestrationGoal,
    agents: list[Agent],
    *,
    loads: dict[uuid.UUID, object] | None = None,
    exclusions: list[dict] | tuple[dict, ...] = (),
    required_work_functions: list[str] | tuple[str, ...] | None = None,
) -> dict:
    """Build the hierarchy semantic input from already-loaded state."""
    loads = loads or {}
    ordered_agents = sorted(agents, key=lambda agent: (agent.name, agent.id))
    return redact_semantic_payload({
        "schema_version": 1,
        "goal": {
            "id": str(goal.id), "objective": goal.objective,
            "success_criteria": goal.success_criteria, "constraints": goal.constraints,
            "weight": goal.weight,
        },
        "selected_manager": manager_from_projection(goal, ordered_agents),
        "required_work_functions": list(required_work_functions or infer_required_work_functions(goal)),
        "agents": [
            {
                "id": str(agent.id), "name": agent.name, "role": agent.role,
                "description": agent.description, "system_prompt": agent.system_prompt,
                "provider": agent.provider, "model": agent.model,
                "adapter_type": agent.adapter_type, "cli_runtime": agent.cli_runtime,
                "capabilities": list(agent.capabilities or []), "config": agent.config or {},
                "workload": (
                    loads[agent.id].to_dict()
                    if agent.id in loads and hasattr(loads[agent.id], "to_dict")
                    else loads.get(agent.id, {
                        "active_tasks": 0, "active_sessions": 0, "outcome_hint_count": 0, "penalty": 0,
                    })
                ),
            }
            for agent in ordered_agents
        ],
        "resolved_proposal_exclusions": list(exclusions),
    })


def resolved_proposal_exclusions_from_decisions(
    decisions: list[OrchestrationAuthorityDecision], *, include_approved: bool = False,
) -> list[dict]:
    exclusions = []
    for decision in decisions:
        if not (
            decision.status == "answered"
            and decision.decision_key.startswith("team_hierarchy:agent:")
            and decision.selected_option in ({"approve", "edit", "reject"} if include_approved else {"edit", "reject"})
        ):
            continue
        option = next((item for item in decision.options if item.get("key") == decision.selected_option), None)
        proposal = option.get("proposed_agent") if isinstance(option, dict) else None
        definition = proposal.get("definition") if isinstance(proposal, dict) else None
        if not isinstance(definition, dict):
            continue
        canonical_definition = AgentCreate.model_validate(definition).model_dump(mode="json")
        canonical = json.dumps(canonical_definition, sort_keys=True, separators=(",", ":"))
        exclusions.append({
            "action": decision.selected_option,
            "definition": canonical_definition,
            "fingerprint": hashlib.sha256(canonical.encode()).hexdigest(),
        })
    return exclusions


def hierarchy_fingerprint_from_projection(payload: dict, analysis: dict | None) -> str:
    stripped = TeamHierarchyProcess._strip_volatile(payload)
    return semantic_fingerprint_from_projection(
        {"input": stripped, "analysis": analysis} if isinstance(analysis, dict) else stripped
    )


def trivial_hierarchy_fingerprint_from_projection(goal: OrchestrationGoal, proposal: dict) -> str:
    payload = {
        "weight": goal.weight,
        "objective": goal.objective,
        "success_criteria": goal.success_criteria,
        "constraints": goal.constraints,
        "authority_model": goal.authority_model,
        "manager_agent_id": str(goal.manager_agent_id) if goal.manager_agent_id else None,
        "manager_user_id": str(goal.manager_user_id) if goal.manager_user_id else None,
        "source_agent_review_process_id": proposal["source_agent_review_process_id"],
        "review_ids": proposal["review_ids"],
        "required_work_functions": proposal["required_work_functions"],
        "role_to_agent": proposal["role_to_agent"],
        "candidate_agent_ids_by_function": {
            work_function: sorted(set(proposal["candidate_agents"].get(work_function, [])))
            for work_function in proposal["required_work_functions"]
        },
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode()).hexdigest()


def infer_required_work_functions(goal: OrchestrationGoal) -> list[str]:
    """Infer a stable minimum hierarchy from the goal's durable definition."""
    if goal.weight == "trivial":
        return []
    from huddleroom.services.orchestration_goal_definition import goal_objective_with_clarifications

    text = json.dumps(
        {
            "objective": goal_objective_with_clarifications(goal),
            "success_criteria": goal.success_criteria,
            "constraints": goal.constraints,
        },
        sort_keys=True,
        default=str,
    ).lower()
    # Constraints are excluded here: roster boilerplate such as "Do not create agents" must not match.
    artifact_text = json.dumps(
        {
            "objective": goal_objective_with_clarifications(goal),
            "success_criteria": goal.success_criteria,
        },
        sort_keys=True,
        default=str,
    ).lower()
    required = {"planning", "summarization"}
    for work_function in _SPECIALIST_FUNCTIONS:
        if any(
            re.search(rf"\b{re.escape(keyword)}\w*\b", text)
            for keyword in PROFILES[work_function].keywords
        ):
            required.add(work_function)
    if goal.weight == "substantial":
        required.add("validation")
    if requires_independent_verification(
        goal.success_criteria, goal_objective_with_clarifications(goal)
    ):
        required.update(_VERIFIER_FUNCTIONS)
    # ponytail: verb heuristic local to inference; shared PROFILES keywords untouched (roster scoring uses them).
    # no add/update/edit/author: they match authority/address/roadmap-update goals
    if _ARTIFACT_VERB_RE.search(artifact_text):
        required.add("implementation")
    if not required.intersection(_SPECIALIST_FUNCTIONS):
        required.add("implementation")
    return [name for name in WORK_FUNCTION_ORDER if name in required]


RECOVERY_TRIGGER_REASON = "approved_recovery"
RECOVERY_APPROVAL = "approved_recovery"
NO_INDEPENDENT_VERIFIER_GAP = "no_independent_verifier"
# Registered recovery gap codes: one uncovered required work function, or an
# independent-verification override -- the gaps the hierarchy already permits.
RECOVERY_GAP_CODES = frozenset((*WORK_FUNCTION_ORDER, NO_INDEPENDENT_VERIFIER_GAP))
_MEMBER_KEYS = ("team_leads", "contributors", "reviewers", "validators", "specialists")


class _RecoveryStrict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _RecoveryAssignment(_RecoveryStrict):
    work_function: str
    agent_id: uuid.UUID

    @field_validator("work_function")
    @classmethod
    def _function(cls, value: str) -> str:
        return _AssignmentV1._function(value)  # pylint: disable=protected-access


class _RecoveryManager(_RecoveryStrict):
    kind: Literal["agent", "human", "none"]
    agent_id: uuid.UUID | None = None
    user_id: uuid.UUID | None = None

    @model_validator(mode="after")
    def _shape(self) -> "_RecoveryManager":
        if (self.agent_id is not None) != (self.kind == "agent") or (self.user_id is not None) != (self.kind == "human"):
            raise ValueError("manager reference does not match its kind")
        return self


class _RecoveryGap(_RecoveryStrict):
    code: str = Field(min_length=1, max_length=255)
    rationale: str = Field(min_length=1, max_length=4_000)


class RecoveryHierarchyAcceptanceV1(_RecoveryStrict):
    """Typed input of the recovery-sourced team_hierarchy snapshot."""

    call_id: str = Field(min_length=1, max_length=100)
    canonical_request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    assignments: list[_RecoveryAssignment] = Field(min_length=1, max_length=128)
    manager: _RecoveryManager
    team_leads: list[uuid.UUID] = Field(max_length=128)
    contributors: list[uuid.UUID] = Field(max_length=128)
    reviewers: list[uuid.UUID] = Field(max_length=128)
    validators: list[uuid.UUID] = Field(max_length=128)
    specialists: list[uuid.UUID] = Field(max_length=128)
    documented_gaps: list[_RecoveryGap] = Field(max_length=128)


def recovery_manager_matches(goal: OrchestrationGoal, manager: dict) -> bool:
    """Recovery v1 keeps the selected manager: the sealed roster fingerprints assume it."""
    kind = manager.get("kind")
    if kind == "agent":
        return goal.manager_agent_id is not None and str(goal.manager_agent_id) == str(manager.get("agent_id"))
    if kind == "human":
        return (goal.manager_agent_id is None and goal.manager_user_id is not None
                and str(goal.manager_user_id) == str(manager.get("user_id")))
    return kind == "none" and goal.manager_agent_id is None and goal.manager_user_id is None


def derive_recovery_hierarchy(
    goal: OrchestrationGoal,
    role_to_agent: dict[str, str],
    eligibility: dict[str, list[str]],
    members: dict[str, list[str]],
    documented_gaps: list[dict],
) -> dict:
    """Validate an approved recovery hierarchy and derive its server-side fields.

    Assignments use only agents accepted for that function; required functions are
    covered or documented by a registered gap; producer/verifier independence holds
    or is explicitly overridden; role membership matches the assignments.
    """
    for work_function, agent_id in role_to_agent.items():
        if work_function not in eligibility.get(agent_id, []):
            raise ValueError("unapproved_assignment")
    required = infer_required_work_functions(goal)
    missing = [work_function for work_function in required if work_function not in role_to_agent]
    independent_functions = TeamHierarchyProcess._independent_functions(goal)  # pylint: disable=protected-access
    producers = {agent_id for work_function, agent_id in role_to_agent.items() if work_function not in _VERIFIER_FUNCTIONS}
    possible = all(
        work_function in role_to_agent
        and TeamHierarchyProcess._independent_of(role_to_agent[work_function], producers, goal)  # pylint: disable=protected-access
        for work_function in independent_functions
    )
    codes = [gap["code"] for gap in documented_gaps]
    permitted = set(missing) | (set() if possible else {NO_INDEPENDENT_VERIFIER_GAP})
    if len(set(codes)) != len(codes) or not set(codes) <= RECOVERY_GAP_CODES or not set(codes) <= permitted:
        raise ValueError("invalid_documented_gap")
    if not possible and NO_INDEPENDENT_VERIFIER_GAP not in codes:
        raise ValueError("independent_verification_conflict")
    if permitted - set(codes):
        raise ValueError("undocumented_gap")
    derived = TeamHierarchyProcess._hierarchy(goal, role_to_agent)  # pylint: disable=protected-access
    for key in _MEMBER_KEYS:
        values = members[key]
        if len(set(values)) != len(values):
            raise ValueError("hierarchy_mismatch")
        if key in {"team_leads", "specialists"}:
            if any(value not in eligibility for value in values):
                raise ValueError("hierarchy_mismatch")
        elif set(values) != set(derived[key]):
            raise ValueError("hierarchy_mismatch")
    independent = TeamHierarchyProcess._independent_verification(  # pylint: disable=protected-access
        independent_functions, role_to_agent,
    )
    return {
        "required_work_functions": required,
        "missing_work_functions": missing,
        "independent_verification": {**independent, "possible": possible},
        "responsibility_conflicts": TeamHierarchyProcess._responsibility_conflicts(role_to_agent),  # pylint: disable=protected-access
        "hierarchy": {"manager": TeamHierarchyProcess._manager(goal), **{key: list(members[key]) for key in _MEMBER_KEYS}},  # pylint: disable=protected-access
    }


class TeamHierarchyProcess:
    """Build and persist the current hierarchy proposal."""

    def __init__(self, analyzer: TeamHierarchyAnalyzer | None = None) -> None:
        self.process_service = OrchestrationProcessService()
        self.review_service = OrchestrationAgentReviewService()
        self.roster_mapper = OrchestrationRosterMapper()
        self.decision_service = OrchestrationAuthorityDecisionService()
        self.warning_service = OrchestrationWarningService()
        self.memory_service = OrchestrationMemoryService()
        self.agent_service = AgentService()
        self.analyzer = analyzer or TeamHierarchyAnalyzer()

    async def advance(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
        *,
        manual: bool = False,
    ) -> dict:
        current = await self.process_service.get_current(db, goal.id, PROCESS_TYPE)
        if current is not None and current.status == "skipped":
            await self.handle_skip(db, goal, current)
            return self._summary("skipped")
        if current is None and not (run.baseline_authorized or manual):
            return self._summary("not_started")

        if goal.weight != "trivial":
            return await self._advance_orchestrated(db, goal, run, current, manual=manual)

        proposal = await self.build_proposal(db, goal, run)
        fingerprint = self._fingerprint(goal, proposal)
        if current is not None:
            capture_input_fingerprint(
                current, fingerprint, PROCESS_VERSION,
                allow_initialize=(current.input_snapshot or {}).get("effective_comparison", {}).get("source") == "copied_seed",
            )
        if current is not None and current.status == "completed" and current.process_version != PROCESS_VERSION:
            # The trivial-tier fingerprint formula (goal + proposal) never
            # hashed agent-snapshot fields, so this version bump can't have
            # changed it -- but guard the same way as the orchestrated tier
            # below for safety: only silently re-stamp when nothing has
            # actually drifted. If it has, bump process_version now (so it
            # isn't rechecked every tick) and leave the stored fingerprint
            # stale so the normal stale-inputs path below fires.
            current.process_version = PROCESS_VERSION
            if current.outputs.get("fingerprint") == fingerprint:
                current.outputs = {**(current.outputs or {}), "fingerprint": fingerprint}
            await db.flush()
        if (
            current is not None
            and current.status == "completed"
            and current.outputs.get("fingerprint") == fingerprint
        ):
            return self._summary("completed")

        stored_fingerprint = (
            effective_input_fingerprint(current)
            if current is not None and isinstance(current.input_snapshot, dict)
            else None
        )
        if (
            current is not None
            and current.status == "waiting_decision"
            and stored_fingerprint
            and stored_fingerprint != fingerprint
        ):
            # Stale inputs while a decision is pending: park stays parked,
            # one-time suggestion instead of cancel+restart (item 3).
            if run.phase in ("baseline", "ready"):
                await self.warning_service.suggest_stale_inputs(
                    db,
                    goal.id,
                    process_type=PROCESS_TYPE,
                    process_run_id=current.id,
                    step_label="team hierarchy",
                    run_id=run.id,
                )
            return self._summary("waiting_decision")
        if (
            current is not None
            and current.status == "running"
            and stored_fingerprint
            and stored_fingerprint != fingerprint
            and (manual or run.phase == "baseline")
        ):
            await self._cancel_pending_for_run(db, goal.id, current.id)
            superseded = current
            superseded.superseded_by_id = superseded.id
            await db.flush()
            current = await self.process_service.start_process(
                db,
                goal.id,
                process_type=PROCESS_TYPE,
                trigger_reason="auto-rerun: hierarchy inputs changed",
                run_id=run.id,
                input_snapshot={"fingerprint": fingerprint},
                process_version=PROCESS_VERSION,
            )
            superseded.superseded_by_id = current.id
            await db.flush()
            # Bug #42: concurrent force-start can return a terminal row; short-circuit if not active
            if current.status not in ("running", "waiting_decision"):
                return self._summary(current.status)
        elif current is not None and current.status == "completed":
            if run.phase == "baseline":
                # Baseline phase: stale inputs get a one-time suggestion, no
                # auto-rerun.
                if run.phase in ("baseline", "ready"):
                    await self.warning_service.suggest_stale_inputs(
                        db,
                        goal.id,
                        process_type=PROCESS_TYPE,
                        process_run_id=current.id,
                        step_label="team hierarchy",
                        run_id=run.id,
                    )
                return self._summary("completed")
            # Post-baseline: reruns are user-initiated only (manual=True);
            # the tick never restarts a completed step on stale inputs.
            if not manual:
                # Gate only applies until Start; later it stays silent.
                if run.phase == "ready":
                    await self.warning_service.suggest_stale_inputs(
                        db,
                        goal.id,
                        process_type=PROCESS_TYPE,
                        process_run_id=current.id,
                        step_label="team hierarchy",
                        run_id=run.id,
                    )
                return self._summary("completed")
            current = await self.process_service.start_process(
                db,
                goal.id,
                process_type=PROCESS_TYPE,
                trigger_reason="auto-rerun: hierarchy inputs changed",
                run_id=run.id,
                input_snapshot={"fingerprint": fingerprint},
                process_version=PROCESS_VERSION,
            )
            # Bug #42: concurrent force-start can return a terminal row; short-circuit if not active
            if current.status not in ("running", "waiting_decision"):
                return self._summary(current.status)
        elif current is None:
            current = await self.process_service.start_process(
                db,
                goal.id,
                process_type=PROCESS_TYPE,
                trigger_reason="tick: no team hierarchy on record",
                run_id=run.id,
                input_snapshot={"fingerprint": fingerprint},
                process_version=PROCESS_VERSION,
            )
            # Bug #42: concurrent force-start can return a terminal row; short-circuit if not active
            if current.status not in ("running", "waiting_decision"):
                return self._summary(current.status)
        elif not stored_fingerprint:
            set_effective_input_fingerprint(current, fingerprint, PROCESS_VERSION, "input_initialized")
            await db.flush()

        agent_definition_review_skipped = await self._agent_definition_review_was_skipped(
            db, goal.id
        )
        if goal.weight == "trivial":
            await self._write_memory(db, goal, run, proposal)
            await self._complete(
                db,
                goal,
                current,
                proposal,
                fingerprint,
                None,
                agent_definitions_reviewed=not agent_definition_review_skipped,
            )
            return self._summary("completed")

        gaps = self._gaps(proposal, agent_definition_review_skipped)
        routing_gaps = self._routing_gaps(proposal, agent_definition_review_skipped)
        standard_eligible = goal.weight == "standard" and not (
            set(routing_gaps) & _STANDARD_TIER_BLOCKING_GAPS
        )
        if standard_eligible and await self._park_on_ineligible_assignment(
            db, goal, run, current, proposal
        ):
            return self._summary("waiting_decision")
        warnings = await self._write_warnings(db, goal, run, current, proposal)
        await self._write_memory(db, goal, run, proposal)
        if standard_eligible:
            await self._complete(
                db,
                goal,
                current,
                proposal,
                fingerprint,
                None,
                warnings=warnings,
                agent_definitions_reviewed=not agent_definition_review_skipped,
            )
            return self._summary("completed")

        self._persist_waiting_outputs(current, proposal, fingerprint, warnings)
        await db.flush()
        waiting_proposal = current.outputs["proposal"]
        decisions = [
            decision
            for decision in await self.decision_service.list_decisions(db, goal.id)
            if decision.decision_key == APPROVAL_DECISION_KEY
        ]
        pending = next(
            (
                decision
                for decision in reversed(decisions)
                if decision.status == "pending"
                and decision.source_process_run_id == current.id
            ),
            None,
        )
        answered = next(
            (
                decision
                for decision in reversed(decisions)
                if decision.status == "answered"
                and decision.source_process_run_id == current.id
            ),
            None,
        )
        if answered is not None and not await self._principal_is_current(db, answered):
            answered = None

        if answered is not None:
            if answered.selected_option == "request_changes":
                current.outputs = {**current.outputs, "change_requested": True}
                await db.flush()
                await self.process_service.park_process(db, current)
                return self._summary("waiting_decision")
            if answered.selected_option == "escalate_to_human":
                questions_created = 0
                if pending is None:
                    await self._ask(
                        db,
                        goal,
                        run,
                        current,
                        waiting_proposal,
                        gaps=gaps,
                        routing_gaps=routing_gaps,
                        force_human=True,
                    )
                    questions_created = 1
                await self.process_service.park_process(db, current)
                return self._summary("waiting_decision", questions_created)
            if answered.selected_option in {"approve", "approve_with_documented_gaps"}:
                selected = next(
                    (
                        option
                        for option in answered.options
                        if option["key"] == answered.selected_option
                    ),
                    None,
                )
                if selected is not None and isinstance(selected.get("proposal"), dict):
                    if await self._park_on_ineligible_assignment(
                        db, goal, run, current, selected["proposal"]
                    ):
                        return self._summary("waiting_decision")
                    if current.status == "waiting_decision":
                        await self.process_service.resume_process(db, current)
                    await self._complete(
                        db,
                        goal,
                        current,
                        selected["proposal"],
                        fingerprint,
                        answered,
                        warnings=warnings,
                        agent_definitions_reviewed=not agent_definition_review_skipped,
                    )
                    return self._summary("completed")

        questions_created = 0
        if pending is None:
            await self._ask(
                db, goal, run, current, waiting_proposal, gaps=gaps, routing_gaps=routing_gaps
            )
            questions_created = 1
        await self.process_service.park_process(db, current)
        return self._summary("waiting_decision", questions_created)

    async def _advance_orchestrated(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
        current: OrchestrationProcessRun | None,
        *,
        manual: bool = False,
    ) -> dict:
        payload = await self._analysis_input(db, goal)
        input_fingerprint = self._semantic_fingerprint(self._strip_volatile(payload))
        if current is not None:
            capture_input_fingerprint(
                current, input_fingerprint, PROCESS_VERSION,
                allow_initialize=(current.input_snapshot or {}).get("effective_comparison", {}).get("source") == "copied_seed",
            )
        if current is not None and current.status in {"running", "waiting_decision"}:
            reserved = await self._apply_reserved_agents(db, goal, run, current)
            if reserved:
                # Reservation can roster-create an approved hierarchy agent in
                # this same continuation. Rebuild before any rerun prompt so
                # it judges the actual accepted definition, never the stale
                # pre-handoff roster projection.
                payload = await self._analysis_input(db, goal)
                input_fingerprint = self._semantic_fingerprint(self._strip_volatile(payload))
            snapshot = current.input_snapshot or {}
            unstamped = reserved - set(snapshot.get("stamped_agent_ids") or ())
            if (
                unstamped
                and effective_input_fingerprint(current) != input_fingerprint
                and effective_input_fingerprint(current) == self._semantic_fingerprint(self._strip_volatile({
                    **payload, "agents": [agent for agent in payload["agents"] if agent["id"] not in unstamped],
                }))
            ):
                # The only drift is the agents this process's approvals created
                # roster-first after the goal lock was released: re-stamp.
                set_effective_input_fingerprint(current, input_fingerprint, PROCESS_VERSION, "approved_agents")
                current.input_snapshot = {**current.input_snapshot, "stamped_agent_ids": sorted(reserved)}
                await db.flush()
        if current is not None and current.status == "completed" and current.process_version != PROCESS_VERSION:
            # A PROCESS_VERSION bump absorbs a fingerprint-*formula* change
            # (dropping provider/model/adapter_type/cli_runtime/config from
            # the hash). It must NOT also silently absorb real drift that
            # happens to land on the same deploy. Recompute what the OLD
            # formula (only workload stripped) would have produced against
            # CURRENT state -- only if that matches what's stored is this a
            # pure formula change.
            old_formula_fingerprint = self._semantic_fingerprint(
                self._strip_volatile(payload, extra_fields=())
            )
            stored_fingerprint = (
                effective_input_fingerprint(current)
                if isinstance(current.input_snapshot, dict)
                else None
            )
            pure_version_bump = stored_fingerprint == old_formula_fingerprint
            current.process_version = PROCESS_VERSION
            if pure_version_bump:
                set_effective_input_fingerprint(current, input_fingerprint, PROCESS_VERSION, "formula_upgrade")
            # else: genuine drift co-occurring with the version bump -- the
            # version is bumped now so it isn't rechecked every tick, but the
            # stored fingerprint is left stale so the normal stale-inputs
            # path below fires (suggestion in baseline / no rerun
            # post-baseline unless user-initiated).
            await db.flush()
        if (
            current is not None and current.status == "completed"
            and effective_input_fingerprint(current) == input_fingerprint
            and current.outputs.get("approval") in {"approve", "approve_with_documented_gaps", RECOVERY_APPROVAL}
        ):
            return self._summary("completed")
        if current is not None and current.status == "completed":
            if run.phase == "baseline":
                # Baseline phase: stale inputs get a one-time suggestion, no
                # auto-rerun.
                if run.phase in ("baseline", "ready"):
                    await self.warning_service.suggest_stale_inputs(
                        db,
                        goal.id,
                        process_type=PROCESS_TYPE,
                        process_run_id=current.id,
                        step_label="team hierarchy",
                        run_id=run.id,
                    )
                return self._summary("completed")
            # Post-baseline: reruns are user-initiated only (manual=True);
            # the tick never restarts a completed step on stale inputs.
            if not manual:
                # Gate only applies until Start; later it stays silent.
                if run.phase == "ready":
                    await self.warning_service.suggest_stale_inputs(
                        db,
                        goal.id,
                        process_type=PROCESS_TYPE,
                        process_run_id=current.id,
                        step_label="team hierarchy",
                        run_id=run.id,
                    )
                return self._summary("completed")
            current = await self.process_service.start_process(
                db, goal.id, process_type=PROCESS_TYPE,
                trigger_reason="tick: orchestrate team hierarchy",
                run_id=run.id, input_snapshot={"fingerprint": input_fingerprint},
                process_version=PROCESS_VERSION,
            )
        if current is None:
            current = await self.process_service.start_process(
                db, goal.id, process_type=PROCESS_TYPE,
                trigger_reason="tick: orchestrate team hierarchy",
                run_id=run.id, input_snapshot={"fingerprint": input_fingerprint},
                process_version=PROCESS_VERSION,
            )
        elif (
            current.status == "waiting_decision"
            and effective_input_fingerprint(current) != input_fingerprint
            and not pending_rerun_intent(current)
        ):
            # Stale inputs while a decision is pending: park stays parked,
            # one-time suggestion instead of cancel+restart (item 3). The
            # explicit continuation intent is excluded and falls through below.
            if run.phase in ("baseline", "ready"):
                await self.warning_service.suggest_stale_inputs(
                    db,
                    goal.id,
                    process_type=PROCESS_TYPE,
                    process_run_id=current.id,
                    step_label="team hierarchy",
                    run_id=run.id,
                )
            return self._summary("waiting_decision")
        elif (effective_input_fingerprint(current) != input_fingerprint or pending_rerun_intent(current)) and (
            manual
            or run.phase == "baseline"
            or pending_rerun_intent(current)
        ):
            await self._cancel_pending_for_run(db, goal.id, current.id)
            revision_context = (current.outputs or {}).get("revision_context")
            resolved_validation_repair_attempted = (current.outputs or {}).get("resolved_validation_repair_attempted")
            current = await self.process_service.start_process(
                db, goal.id, process_type=PROCESS_TYPE,
                trigger_reason="auto-rerun: hierarchy inputs changed",
                run_id=run.id, input_snapshot={"fingerprint": input_fingerprint},
                process_version=PROCESS_VERSION,
                supersede_waiting=True,
                supersede_running=True,
            )
            if isinstance(revision_context, dict) or resolved_validation_repair_attempted:
                current.outputs = {
                    **({"revision_context": revision_context} if isinstance(revision_context, dict) else {}),
                    **({"resolved_validation_repair_attempted": True} if resolved_validation_repair_attempted else {}),
                }
        if current.status not in {"running", "waiting_decision"}:
            return self._summary(current.status)

        analysis = current.outputs.get("analysis") if isinstance(current.outputs, dict) else None
        if not isinstance(analysis, dict):
            from huddleroom.models.project import Project
            _proj = await db.get(Project, goal.project_id)
            project_dict = {"name": _proj.name, "description": _proj.description} if _proj else None
            revision_context = (current.outputs or {}).get("revision_context")
            prompt_payload = await self._hierarchy_prompt_payload(
                db, goal, payload, revision_context if isinstance(revision_context, dict) else None
            )
            try:
                result = await self.analyzer.review(
                    prompt_payload, project=project_dict, project_id=goal.project_id,
                )
            except Exception as exc:  # analyzer normalizes provider and semantic failures
                safe_error = _safe_completion_error(exc)
                request = getattr(exc, "request", None) or TeamHierarchyAnalyzer.build_request(
                    prompt_payload, project=project_dict
                )
                current.outputs = {
                    **({"revision_context": revision_context} if isinstance(revision_context, dict) else {}),
                    **({"resolved_validation_repair_attempted": True}
                       if (current.outputs or {}).get("resolved_validation_repair_attempted") else {}),
                    "_lm_retry": {"kind": PROCESS_TYPE, "version": 1, "request": redact_semantic_payload(request)},
                    "retryable": True,
                    "error": safe_error,
                }
                await db.flush()
                return {**self._summary(current.status), "retryable": True, "error": safe_error}
            analysis = result.to_dict()
            if isinstance(revision_context, dict) and not revision_context.get("feedback_items"):
                revision_context = {
                    **revision_context,
                    "feedback_items": [
                        {"item_id": item["item_id"], "source_excerpt": item["source_excerpt"]}
                        for item in analysis.get("feedback_results", [])
                    ],
                }
            current.outputs = {
                **({"revision_context": revision_context} if isinstance(revision_context, dict) else {}),
                **({"resolved_validation_repair_attempted": True}
                   if (current.outputs or {}).get("resolved_validation_repair_attempted") else {}),
                "analysis": analysis,
                "fingerprint": self._semantic_fingerprint({"input": self._strip_volatile(payload), "analysis": analysis}),
            }
            await db.flush()

        if (
            (current.outputs or {}).get("resolved_validation_repair_attempted")
            and (
                isinstance((current.outputs or {}).get("revision_context"), dict)
                or (current.outputs or {}).get("resolved_validation_retry_repair")
            )
            and analysis.get("proposed_agents")
        ):
            current.outputs = {
                **current.outputs,
                "retryable": False,
                "semantic_error": "resolved hierarchy repair must return final covered or gap evidence",
            }
            await self.process_service.park_process(db, current)
            await db.flush()
            return self._summary("waiting_decision")

        proposal_decisions = [
            decision for decision in await self.decision_service.list_decisions(db, goal.id)
            if decision.source_process_run_id == current.id
            and decision.decision_key.startswith("team_hierarchy:agent:")
        ]
        proposal_by_key = {
            f"team_hierarchy:agent:{proposal['proposal_id']}": proposal
            for proposal in analysis["proposed_agents"]
        }
        created_by_ref: dict[str, str] = {}
        for decision in proposal_decisions:
            if decision.status != "answered" or decision.selected_option != "approve":
                continue
            metadata = self._decision_metadata(decision.reason)
            created_id = metadata.get("created_agent_id")
            if created_id is None:
                # Reserve the id under the goal lock; the row is created roster-first.
                AgentService.validate_create(AgentCreate(**proposal_by_key[decision.decision_key]["definition"]))
                created_id = str(uuid.uuid4())
                metadata["created_agent_id"] = created_id
                await db.execute(update(OrchestrationAuthorityDecision).where(
                    OrchestrationAuthorityDecision.id == decision.id
                ).values(reason=json.dumps(metadata, sort_keys=True)))
                decision.reason = json.dumps(metadata, sort_keys=True)
            from huddleroom.services.orchestration_answer_receipts import record_current_answer_effect
            await record_current_answer_effect(db, f"created_agent_id:{decision.id}", "agent", uuid.UUID(created_id))
            created_by_ref[f"proposal:{decision.decision_key.rsplit(':', 1)[-1]}"] = created_id

        if created_by_ref:
            await self._apply_reserved_agents(db, goal, run, current)
            set_effective_input_fingerprint(
                current, self._semantic_fingerprint(self._strip_volatile(await self._analysis_input(db, goal))),
                PROCESS_VERSION, "approved_agents",
            )
            current.input_snapshot = {**current.input_snapshot, "stamped_agent_ids": sorted(created_by_ref.values())}

        if len(proposal_decisions) == len(proposal_by_key) and all(
            decision.status == "answered" and decision.selected_option == "approve"
            for decision in proposal_decisions
        ):
            revision_context = (current.outputs or {}).get("revision_context")
            frozen = self._resolve_analysis(analysis, created_by_ref)
            refreshed_payload = await self._analysis_input(db, goal)
            try:
                refreshed_prompt = await self._hierarchy_prompt_payload(
                    db, goal, refreshed_payload, revision_context if isinstance(revision_context, dict) else None
                )
                parse_team_hierarchy_analysis({
                    "proposed_agents": [],
                    "assignments": frozen["assignments"],
                    "reporting_lines": frozen["reporting_lines"],
                    "documented_gaps": frozen["documented_gaps"],
                    "rationale": frozen["rationale"],
                    "self_review": frozen["self_review"],
                    "feedback_results": frozen["feedback_results"],
                    "responsibility_coverage": frozen["responsibility_coverage"],
                }, refreshed_prompt)
            except ValueError as exc:
                if (current.outputs or {}).get("resolved_validation_repair_attempted"):
                    current.outputs = {**current.outputs, "retryable": False, "semantic_error": str(exc)}
                    await self.process_service.park_process(db, current)
                    await db.flush()
                    return self._summary("waiting_decision")
                current.outputs = {**current.outputs, "retryable": False, "semantic_error": str(exc)}
                current.outputs["resolved_validation_repair_attempted"] = True
                await self._cancel_pending_for_run(db, goal.id, current.id)
                request_process_rerun(current, "resolved_validation_failed")
                await db.flush()
                return await self._advance_orchestrated(db, goal, run, current)
            approval = next((
                decision for decision in reversed(await self.decision_service.list_decisions(db, goal.id))
                if decision.source_process_run_id == current.id
                and decision.decision_key == APPROVAL_DECISION_KEY
                and decision.status in {"pending", "answered"}
            ), None)
            if (
                approval is not None
                and approval.status == "answered"
                and not await self._principal_is_current(db, approval)
            ):
                approval = None
            if approval is not None and approval.status == "answered" and approval.selected_option in {
                "approve", "approve_with_documented_gaps"
            }:
                selected = next(option for option in approval.options if option["key"] == approval.selected_option)
                selected_proposal = selected["proposal"]
                substantive_gap_ids = sorted(
                    item["responsibility_id"] for item in selected_proposal.get("responsibility_coverage", [])
                    if item.get("status") == "gap"
                )
                has_gaps = bool(substantive_gap_ids or selected_proposal["documented_gaps"])
                if has_gaps and approval.selected_option != "approve_with_documented_gaps":
                    await self.process_service.park_process(db, current)
                    return self._summary("waiting_decision")
                outputs = {
                    **selected_proposal,
                    "proposal": selected_proposal,
                    "fingerprint": effective_input_fingerprint(current),
                    "approval": approval.selected_option,
                    "approval_decision_id": str(approval.id),
                    "accepted_responsibility_gap_ids": substantive_gap_ids,
                    **({"revision_context": revision_context} if isinstance(revision_context, dict) else {}),
                    "gates": {
                        "team_structure_reviewed": True,
                        "required_work_functions_mapped": not has_gaps,
                        "agent_definitions_reviewed": True,
                        "missing_capabilities_handled": has_gaps or not selected_proposal["documented_gaps"],
                        "independent_verification_possible_or_overridden": True,
                    },
                }
                if await self._park_on_ineligible_assignment(db, goal, run, current, outputs):
                    return self._summary("waiting_decision")
                if current.status == "waiting_decision":
                    await self.process_service.resume_process(db, current)
                await self.process_service.complete_process(db, current, outputs=outputs)
                return self._summary("completed")
            if approval is not None and approval.status == "answered":
                replacement_revision = self._revision_context(current, approval)
                prior_revision = (current.outputs or {}).get("revision_context")
                replacement_outputs = {
                    **(current.outputs or {}),
                    "revision_context": replacement_revision,
                }
                if not (
                    isinstance(prior_revision, dict)
                    and prior_revision.get("source_decision_id") == replacement_revision["source_decision_id"]
                ):
                    replacement_outputs.pop("resolved_validation_repair_attempted", None)
                    replacement_outputs.pop("resolved_validation_retry_repair", None)
                current.outputs = replacement_outputs
                await self._cancel_pending_for_run(db, goal.id, current.id)
                request_process_rerun(current, "hierarchy_changes_requested", approval.id)
                await db.flush()
                return await self._advance_orchestrated(db, goal, run, current)
            if approval is None:
                gaps = list(frozen["documented_gaps"])
                substantive_gaps = [
                    item["responsibility_id"] for item in frozen.get("responsibility_coverage", [])
                    if item.get("status") == "gap"
                ]
                approve_key = "approve_with_documented_gaps" if gaps or substantive_gaps else "approve"
                current.outputs = {
                    **(current.outputs or {}),
                    "proposal": frozen,
                    "gates": {
                        "required_work_functions_mapped": not frozen["documented_gaps"] and not substantive_gaps,
                        "missing_capabilities_handled": not frozen["documented_gaps"] and not substantive_gaps,
                    },
                }
                approval_decision = await self.decision_service.create_pending(
                    db, goal.id, decision_key=APPROVAL_DECISION_KEY,
                    title="Approve team hierarchy",
                    question="Approve the proposed team hierarchy and role assignments?",
                    authority="human", authority_agent_id=None,
                    options=[{"key": approve_key, "proposal": frozen}, {"key": "request_changes"}],
                    context=json.dumps({"documented_gaps": frozen["documented_gaps"], "responsibility_gap_ids": substantive_gaps}, sort_keys=True),
                    recommendation=approve_key, run_id=run.id, source_process_run_id=current.id,
                )
                questions_created = 1
            else:
                questions_created = 0
                approval_decision = approval
            from huddleroom.services.orchestration_answer_receipts import record_current_answer_effect
            await record_current_answer_effect(
                db, f"continuation_decision:{current.id}", "decision", approval_decision.id,
            )
            await self.process_service.park_process(db, current)
            return self._summary("waiting_decision", questions_created)

        questions_created = 0
        existing_keys = {
            decision.decision_key
            for decision in await self.decision_service.list_decisions(db, goal.id)
            if decision.source_process_run_id == current.id
        }
        for proposal in analysis["proposed_agents"]:
            decision_key = f"team_hierarchy:agent:{proposal['proposal_id']}"
            if decision_key not in existing_keys:
                await self.decision_service.create_pending(
                    db, goal.id, decision_key=decision_key,
                    title=f"Review proposed agent: {proposal['definition']['name']}",
                    question="Approve, edit, or reject this complete proposed agent definition?",
                    authority="human",
                    options=[
                        {"key": "approve", "proposed_agent": proposal},
                        {"key": "edit", "proposed_agent": proposal},
                        {"key": "reject", "proposed_agent": proposal},
                    ],
                    context=json.dumps({"proposal": proposal}, sort_keys=True),
                    recommendation="approve",
                    consequences="Approved or edited definitions are applied; rejection creates no agent.",
                    run_id=run.id,
                    source_process_run_id=current.id,
                )
                questions_created += 1
        await self.process_service.park_process(db, current)
        return self._summary("waiting_decision", questions_created)

    async def _park_on_ineligible_assignment(
        self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun,
        current: OrchestrationProcessRun, proposal: dict,
    ) -> bool:
        """True (and parked behind the stale-inputs suggestion) when an accepted
        assignment is not covered by the current agent-definition eligibility."""
        from huddleroom.services.orchestration_agent_definition_review import accepted_work_functions

        accepted = await accepted_work_functions(db, goal.id)
        if accepted is None:
            return False
        pairs = (
            [(row["agent_ref"], row["work_function"]) for row in proposal["assignments"]]
            if "assignments" in proposal
            else [(agent_id, work_function) for work_function, agent_id in proposal.get("role_to_agent", {}).items()]
        )
        review = await self.process_service.get_current(db, goal.id, "agent_definition_review")
        considered: dict[str, set[str]] = {}  # functions the review actually evaluated, per agent
        for item in await self.review_service.list_reviews(db, goal.id):
            if review is not None and item.source_process_run_id == review.id and item.agent_id is not None:
                considered.setdefault(str(item.agent_id), set()).update(item.proposed_work_functions or [])
        # ponytail: functions the review never evaluated are not conflicts (select_plan_author still gates them)
        if not any(
            work_function in considered.get(str(agent_ref), ())
            and work_function not in (accepted.get(str(agent_ref)) or [])
            for agent_ref, work_function in pairs
        ):
            return False
        await self.warning_service.suggest_stale_inputs(
            db, goal.id, process_type=PROCESS_TYPE, process_run_id=current.id,
            step_label="agent eligibility", run_id=run.id,
        )
        await self.process_service.park_process(db, current)
        return True

    async def _apply_reserved_agents(
        self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun, current: OrchestrationProcessRun,
    ) -> set[str]:
        """Create reserved-but-missing approved agents roster-first; return reserved ids.

        Approve/edit answers record ``created_agent_id`` under the goal lock. The
        row is written by the roster coordinator (handoff), so a crash or stale
        apply leaves a reservation this method reconstructs idempotently.
        """
        from huddleroom.services.agent_service import hand_off_roster_delta
        from huddleroom.services.orchestration_roster_mutation import ValidatedRosterDelta

        analysis = current.outputs.get("analysis") if isinstance(current.outputs, dict) else None
        proposals = {
            f"team_hierarchy:agent:{proposal['proposal_id']}": proposal
            for proposal in (analysis or {}).get("proposed_agents", [])
        }
        reserved, missing, decision_ids = set(), [], []
        for decision in await self.decision_service.list_decisions(db, goal.id):
            if (
                decision.source_process_run_id != current.id
                or not decision.decision_key.startswith("team_hierarchy:agent:")
                or decision.status != "answered"
                or decision.selected_option not in {"approve", "edit"}
            ):
                continue
            metadata = self._decision_metadata(decision.reason)
            created_id = metadata.get("created_agent_id")
            if not isinstance(created_id, str):
                continue
            reserved.add(created_id)
            if await db.get(Agent, uuid.UUID(created_id)) is not None:
                continue
            definition = (
                metadata.get("edited_agent") if decision.selected_option == "edit"
                else (proposals.get(decision.decision_key) or {}).get("definition")
            )
            if not isinstance(definition, dict):
                continue  # legacy answer created inline; nothing to reconstruct
            data = AgentCreate(**definition)
            AgentService.validate_create(data)
            missing.append(AgentService.state_from_create(data, uuid.UUID(created_id)))
            decision_ids.append(decision.id)
        if missing:
            await hand_off_roster_delta(
                db, ValidatedRosterDelta(creates=tuple(missing)), goal_id=goal.id, run_id=run.id,
                process_id=current.id, decision_ids=decision_ids,
            )
        return reserved

    @staticmethod
    def _decision_metadata(reason: str | None) -> dict:
        if not reason:
            return {}
        try:
            value = json.loads(reason)
        except (TypeError, ValueError):
            return {"reason": reason}
        return value if isinstance(value, dict) else {"reason": reason}

    @staticmethod
    def _resolve_analysis(analysis: dict, created_by_ref: dict[str, str]) -> dict:
        def resolve(value: str) -> str:
            return created_by_ref.get(value, value)

        assignments = [
            {**row, "agent_ref": resolve(row["agent_ref"])}
            for row in analysis["assignments"]
        ]
        reporting_lines = [
            {
                **row,
                "agent_ref": resolve(row["agent_ref"]),
                "reports_to": resolve(row["reports_to"]),
            }
            for row in analysis["reporting_lines"]
        ]
        used = {row["agent_ref"] for row in assignments}
        approved = set(created_by_ref.values())
        return {
            "assignments": assignments,
            "reporting_lines": reporting_lines,
            "documented_gaps": list(analysis["documented_gaps"]),
            "rationale": analysis["rationale"],
            "self_review": analysis["self_review"],
            "feedback_results": list(analysis.get("feedback_results", [])),
            "responsibility_coverage": list(analysis.get("responsibility_coverage", [])),
            "approved_but_unused_agent_ids": sorted(approved - used),
        }

    def _revision_context(
        self, current: OrchestrationProcessRun, decision: OrchestrationAuthorityDecision,
    ) -> dict:
        """Freeze a request-change answer once, before its successor is drafted."""
        existing = (current.outputs or {}).get("revision_context")
        if isinstance(existing, dict) and existing.get("source_decision_id") == str(decision.id):
            return existing
        if isinstance(current.outputs, dict):
            current.outputs = {
                key: value for key, value in current.outputs.items()
                if key not in {"resolved_validation_repair_attempted", "resolved_validation_retry_repair"}
            }
        saved_reason = self._decision_metadata(decision.reason).get("reason", decision.reason) or ""
        feedback_text = str(saved_reason)
        if not feedback_text.strip():
            raise ValueError("team hierarchy request_changes requires feedback")
        previous = next(
            (option.get("proposal") for option in decision.options
             if isinstance(option, dict) and option.get("key") == "request_changes" and option.get("proposal")),
            (current.outputs or {}).get("proposal"),
        )
        return {
            "source_process_id": str(current.id),
            "source_decision_id": str(decision.id),
            "saved_reason": saved_reason,
            "feedback_text": feedback_text,
            "previous_proposal": previous,
            "feedback_items": [],
        }

    async def _analysis_input(self, db: AsyncSession, goal: OrchestrationGoal) -> dict:
        agents = list((await db.execute(
            select(Agent).where(Agent.is_active.is_(True)).order_by(Agent.name, Agent.id)
        )).scalars())
        loads = await self.roster_mapper.loads_by_agent(db, goal.project_id)
        exclusions = resolved_proposal_exclusions_from_decisions(
            await self.decision_service.list_decisions(db, goal.id),
        )
        return hierarchy_analysis_input_from_projection(
            goal, agents, loads=loads, exclusions=exclusions,
        )

    async def _hierarchy_prompt_payload(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        base_payload: dict,
        revision_context: dict | None,
    ) -> dict:
        """Add accepted review evidence for the analyzer without changing input identity."""
        payload = json.loads(json.dumps(base_payload, default=str))
        review_process = await self.process_service.get_current(db, goal.id, "agent_definition_review")
        reviews = []
        decisions = await self.decision_service.list_decisions(db, goal.id)
        if review_process is not None and review_process.status == "completed":
            eligibility = (review_process.outputs or {}).get("eligibility")
            assessments = (review_process.outputs or {}).get("semantic_assessments")
            agents = {agent["id"]: agent for agent in payload.get("agents", [])}
            if isinstance(eligibility, dict):
                accepted_decisions = {}
                for decision in decisions:
                    if (
                        decision.source_process_run_id != review_process.id
                        or decision.status != "answered"
                        or decision.selected_option not in {"approve", "edit"}
                    ):
                        continue
                    try:
                        decision_context = json.loads(decision.context or "{}")
                    except (TypeError, ValueError):
                        continue
                    agent_id = decision_context.get("agent_id") if isinstance(decision_context, dict) else None
                    if isinstance(agent_id, str):
                        accepted_decisions[agent_id] = decision
                for review in await self.review_service.list_reviews(db, goal.id):
                    agent_id = str(review.agent_id) if review.agent_id is not None else None
                    if review.source_process_run_id != review_process.id or agent_id not in eligibility:
                        continue
                    agent = agents.get(agent_id)
                    if agent is None:
                        continue
                    assessment = assessments.get(agent_id, {}) if isinstance(assessments, dict) else {}
                    decision = accepted_decisions.get(agent_id)
                    # A passing review that proposed no change creates no decision but is accepted.
                    recovered = not assessments and bool((review_process.outputs or {}).get("recovery_provenance"))
                    unchanged = assessment.get("status") == "approved"
                    reviews.append({
                        "review_id": str(review.id),
                        "source_process_id": str(review_process.id),
                        "agent_id": agent_id,
                        "agent_name": agent["name"],
                        "approved_work_functions": list(eligibility[agent_id]),
                        "definition_snapshot": review.definition_snapshot,
                        "accepted_definition": {
                            key: agent[key]
                            for key in ("name", "role", "description", "system_prompt", "capabilities")
                        },
                        "assessment": {
                            "reason": assessment.get("reason"),
                            "problems": assessment.get("problems", []),
                            "boundaries": list(review.risks or []),
                        },
                        "accepted_decision_id": str(decision.id) if decision is not None else None,
                        "accepted_decision": decision.selected_option if decision is not None else None,
                        "review_accepted": decision is not None or unchanged or recovered,
                        "acceptance_basis": (
                            "human_decision" if decision is not None
                            else "review_passed_unchanged" if unchanged
                            else "recovery_approved" if recovered else None
                        ),
                    })
        agents = {agent["id"]: agent for agent in payload.get("agents", [])}
        for decision in decisions:
            if not (
                decision.decision_key.startswith("team_hierarchy:agent:")
                and decision.status == "answered"
                and decision.selected_option in {"approve", "edit"}
            ):
                continue
            created_id = self._decision_metadata(decision.reason).get("created_agent_id")
            agent = agents.get(created_id) if isinstance(created_id, str) else None
            if agent is None:
                continue
            try:
                accepted_definition = AgentCreate.model_validate({
                    key: agent[key]
                    for key in (
                        "name", "role", "description", "provider", "model", "system_prompt",
                        "adapter_type", "cli_runtime", "capabilities", "config",
                    )
                }).model_dump(mode="json")
            except (KeyError, ValueError):
                continue
            reviews.append({
                "review_id": f"hierarchy-decision:{decision.id}",
                "source_process_id": str(decision.source_process_run_id),
                "agent_id": created_id,
                "agent_name": agent["name"],
                # The analyzer must rejudge this actual accepted definition; membership
                # here merely keeps its result within the existing generic taxonomy.
                "approved_work_functions": list(payload.get("required_work_functions", [])),
                "definition_snapshot": accepted_definition,
                "accepted_definition": accepted_definition,
                "assessment": {
                    "reason": "Rejudge this approved hierarchy definition for the proposed responsibility.",
                    "problems": [], "boundaries": [],
                },
                "accepted_decision_id": str(decision.id),
                "accepted_decision": decision.selected_option,
                "review_accepted": True,
                "acceptance_basis": "human_decision",
            })
        payload["hierarchy_context"] = {
            "agent_reviews": reviews,
            "revision": revision_context,
        }
        if isinstance(revision_context, dict):
            approved = [
                item for item in resolved_proposal_exclusions_from_decisions(decisions, include_approved=True)
                if item["action"] == "approve"
            ]
            if approved:
                payload["resolved_proposal_exclusions"] = [
                    *payload.get("resolved_proposal_exclusions", []), *approved,
                ]
        return payload

    @staticmethod
    def _semantic_fingerprint(payload: dict) -> str:
        return semantic_fingerprint_from_projection(payload)

    @staticmethod
    def _strip_volatile(
        payload: dict,
        *,
        extra_fields: tuple[str, ...] = ("provider", "model", "adapter_type", "cli_runtime", "config"),
    ) -> dict:
        """Drop per-agent workload (and, by default, runtime/routing) fields
        before fingerprinting -- for the fingerprint only, never for the
        payload sent to the analyzer.

        Workload (active_tasks/active_sessions/etc.) changes on every task or
        session creation and is not a change to hierarchy *inputs* (roster,
        agent definitions, goal definition, exclusions). It must never affect
        staleness fingerprints or every delegation action would falsely
        invalidate an already-approved hierarchy.

        `extra_fields` additionally excludes runtime/routing fields
        (provider/model/adapter_type/cli_runtime/config) by default: changing
        these doesn't change what the hierarchy should look like, so they
        shouldn't trigger a stale-inputs suggestion. Callers doing PROCESS_VERSION
        drift detection pass `extra_fields=()` to reproduce the pre-bump
        (workload-only) formula.
        """
        stripped = json.loads(json.dumps(payload, default=str))
        stripped.pop("hierarchy_context", None)
        stripped["resolved_proposal_exclusions"] = [
            item for item in stripped.get("resolved_proposal_exclusions", [])
            if not isinstance(item, dict) or item.get("action") != "approve"
        ]
        for agent in stripped.get("agents", []):
            agent.pop("workload", None)
            for field in extra_fields:
                agent.pop(field, None)
        return stripped

    async def current_fingerprint(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
    ) -> str:
        if goal.weight != "trivial":
            current = await self.process_service.get_current(db, goal.id, PROCESS_TYPE)
            analysis = current.outputs.get("analysis") if current and isinstance(current.outputs, dict) else None
            payload = await self._analysis_input(db, goal)
            return hierarchy_fingerprint_from_projection(payload, analysis)
        proposal = await self.build_proposal(db, goal, run)
        return self._fingerprint(goal, proposal)

    async def retry_failed(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
        current: OrchestrationProcessRun,
    ) -> dict:
        checkpoint = (current.outputs or {}).get("_lm_retry")
        if not isinstance(checkpoint, dict) or checkpoint.get("kind") != PROCESS_TYPE:
            raise ValueError("team hierarchy has no valid retry checkpoint")
        request = checkpoint.get("request")
        if not isinstance(request, dict):
            raise ValueError("team hierarchy retry request is invalid")
        try:
            payload = json.loads(request["messages"][1]["content"])
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("team hierarchy retry request is invalid") from exc
        hierarchy_context = payload.get("hierarchy_context") if isinstance(payload, dict) else None
        revision_context = hierarchy_context.get("revision") if isinstance(hierarchy_context, dict) else None
        try:
            result = await self.analyzer.review_request(request, project_id=goal.project_id)
        except Exception as exc:  # mirror advance(): re-park instead of surfacing a 500
            safe_error = _safe_completion_error(exc)
            current.outputs = {
                **({"revision_context": revision_context} if isinstance(revision_context, dict) else {}),
                **({"resolved_validation_repair_attempted": True}
                   if (current.outputs or {}).get("resolved_validation_repair_attempted") else {}),
                "_lm_retry": {"kind": PROCESS_TYPE, "version": 1, "request": redact_semantic_payload(request)},
                "retryable": True,
                "error": safe_error,
            }
            await db.flush()
            return {**self._summary(current.status), "retryable": True, "error": safe_error}
        analysis = result.to_dict()
        if isinstance(revision_context, dict) and not revision_context.get("feedback_items"):
            revision_context = {
                **revision_context,
                "feedback_items": [
                    {"item_id": item["item_id"], "source_excerpt": item["source_excerpt"]}
                    for item in analysis.get("feedback_results", [])
                ],
            }
        current.outputs = {
            **({"revision_context": revision_context} if isinstance(revision_context, dict) else {}),
            **({"resolved_validation_repair_attempted": True}
               if (current.outputs or {}).get("resolved_validation_repair_attempted") else {}),
            **({"resolved_validation_retry_repair": True}
               if (current.outputs or {}).get("resolved_validation_repair_attempted") else {}),
            "analysis": analysis,
            "fingerprint": self._semantic_fingerprint({"input": self._strip_volatile(payload), "analysis": analysis}),
        }
        await db.flush()
        return await self.advance(db, goal, run)

    async def handle_skip(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        skipped_run: OrchestrationProcessRun,
    ) -> None:
        await self.warning_service.create_warning(
            db,
            goal.id,
            warning_type="team_hierarchy_not_reviewed",
            severity="warning",
            message=TEAM_HIERARCHY_SKIP_MESSAGE,
            run_id=skipped_run.run_id,
            source_process_run_id=skipped_run.id,
        )
        await self.memory_service.upsert_section(
            db,
            goal.project_id,
            goal.id,
            section_key=MEMORY_SECTION_KEY,
            title="Team hierarchy and role fit",
            body=TEAM_HIERARCHY_SKIP_MESSAGE,
            summary="Team hierarchy review was skipped by the human.",
            toc_order=MEMORY_TOC_ORDER,
            run_id=skipped_run.run_id,
            created_by="orchestrator:team_hierarchy",
        )

    async def persist_recovery_acceptance(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
        *,
        accepted_payload: dict,
        proposal_id: uuid.UUID,
        proposal_hash: str,
        authority_decision_id: uuid.UUID,
        evidence: tuple[EvidenceV1, ...],
    ) -> OrchestrationProcessRun:
        """Write the accepted hierarchy of an approved recovery configuration.

        Derived server-side from the validated assignments against the recovery-sourced
        definition snapshot of the same proposal; no LM analysis, no waiting_decision,
        no hierarchy approval card. The caller owns the transaction.
        """
        payload = RecoveryHierarchyAcceptanceV1.model_validate(accepted_payload)
        evidence = tuple(EvidenceV1.model_validate(item) for item in evidence)
        if goal.weight == "trivial" or run.goal_id != goal.id:
            raise ValueError("unsupported_recovery_goal")
        if not recovery_manager_matches(goal, payload.manager.model_dump(mode="json")):
            raise ValueError("manager_mismatch")
        provenance = await self.process_service.recovery_provenance(
            db, goal.id, proposal_id=proposal_id, proposal_hash=proposal_hash,
            authority_decision_id=authority_decision_id, call_id=payload.call_id,
            canonical_request_hash=payload.canonical_request_hash,
        )
        definition = await self.process_service.get_current(db, goal.id, "agent_definition_review")
        definition_outputs = (definition.outputs or {}) if definition is not None else {}
        if (
            definition is None or definition.status != "completed"
            or (definition_outputs.get("recovery_provenance") or {}).get("proposal_id") != str(proposal_id)
            or (definition_outputs.get("recovery_provenance") or {}).get("proposal_hash") != proposal_hash
        ):
            raise ValueError("recovery_definition_snapshot_missing")
        eligibility = definition_outputs.get("eligibility") or {}
        role_to_agent: dict[str, str] = {}
        for item in payload.assignments:
            if item.work_function in role_to_agent:
                raise ValueError("duplicate_function_assignment")
            role_to_agent[item.work_function] = str(item.agent_id)
        members = {key: [str(value) for value in getattr(payload, key)] for key in _MEMBER_KEYS}
        referenced = set(role_to_agent.values()).union(*members.values())
        agents = {}
        for agent_id in sorted(referenced):
            agent = await db.get(Agent, uuid.UUID(agent_id))
            if agent is None or not agent.is_active:
                raise ValueError("inactive_or_unknown_agent")
            agents[agent_id] = agent
        documented_gaps = [gap.model_dump(mode="json") for gap in payload.documented_gaps]
        derived = derive_recovery_hierarchy(goal, role_to_agent, eligibility, members, documented_gaps)
        previous = await self.process_service.get_current(db, goal.id, PROCESS_TYPE)
        reporting_lines = (
            list((previous.outputs or {}).get("reporting_lines", []))
            if previous is not None and previous.status == "completed" else []
        )

        # -- writes start here --------------------------------------------------------------
        fingerprint = hierarchy_fingerprint_from_projection(await self._analysis_input(db, goal), None)
        current = await self.process_service.start_process(
            db, goal.id, process_type=PROCESS_TYPE, trigger_reason=RECOVERY_TRIGGER_REASON,
            run_id=run.id, input_snapshot={"fingerprint": fingerprint},
            process_version=PROCESS_VERSION, supersede_waiting=True, supersede_running=True,
        )
        if current.trigger_reason != RECOVERY_TRIGGER_REASON or current.status != "running":
            raise ValueError("recovery_snapshot_not_started")
        for decision in await self.decision_service.list_decisions(db, goal.id, status="pending"):
            if decision.decision_key.startswith(f"{PROCESS_TYPE}:"):
                await self.decision_service.cancel_decision(
                    db, decision, reason="superseded by an approved recovery configuration",
                )
        weak_fits = [
            {"work_function": work_function, "agent_id": agent_id}
            for work_function, agent_id in role_to_agent.items()
            if work_function in PROFILES
            and self.roster_mapper.score_agent_definition(agents[agent_id], work_function).score < WEAK_FIT_THRESHOLD
        ]
        proposal = {**derived, "role_to_agent": role_to_agent, "weak_fits": weak_fits}
        warnings = await self._write_warnings(db, goal, run, current, proposal)
        overridden = not derived["independent_verification"]["possible"]
        if overridden:
            derived["independent_verification"]["overridden_by_decision_id"] = str(authority_decision_id)
        outputs = {
            "source_agent_review_process_id": str(definition.id),
            "review_ids": sorted(str(item) for item in definition_outputs.get("review_ids", [])),
            "weight": goal.weight,
            "compressed": goal.weight != "substantial",
            **derived,
            "role_to_agent": role_to_agent,
            "assignments": [{"work_function": item.work_function, "agent_ref": str(item.agent_id)}
                            for item in payload.assignments],
            "reporting_lines": reporting_lines,
            "documented_gaps": documented_gaps,
            "weak_fits": weak_fits,
            "suggestions": [],
            "fingerprint": fingerprint,
            "approval": RECOVERY_APPROVAL,
            "approval_decision_id": str(authority_decision_id),
            "warning_ids": [str(warning.id) for warning in warnings],
            "gates": self._gates(
                missing=bool(derived["missing_work_functions"]),
                possible=bool(derived["independent_verification"]["possible"]),
                documented_gaps=bool(documented_gaps),  # validated: every gap is documented
                agent_definitions_reviewed=True,
            ),
            "recovery_provenance": {**provenance, "evidence_labels": [item.label for item in evidence]},
        }
        assignments_text = ", ".join(f"{fn}: agent:{agent_id}" for fn, agent_id in role_to_agent.items()) or "none"
        independent = derived["independent_verification"]
        await self.memory_service.upsert_section(
            db, goal.project_id, goal.id, section_key=MEMORY_SECTION_KEY, title="Team hierarchy and role fit",
            body="\n".join((
                f"Accepted through approved recovery (proposal {proposal_id}); no standard hierarchy review was run.",
                f"Required work functions: {', '.join(derived['required_work_functions']) or 'none'}",
                f"Assignments: {assignments_text}",
                f"Documented gaps: {', '.join(gap['code'] for gap in documented_gaps) or 'none'}",
                "Independent verification: "
                f"required={independent['required']}, possible={independent['possible']}, overridden={overridden}",
            )),
            summary=(
                f"Approved recovery mapped {len(role_to_agent)} of "
                f"{len(derived['required_work_functions'])} required work functions."
            ),
            toc_order=MEMORY_TOC_ORDER, run_id=run.id, created_by="orchestrator:team_hierarchy",
        )
        await self._resolve_prior_skip_warnings(
            db, goal.id, current.id, all_types=True,
            reason="superseded by an approved recovery configuration",
        )
        return await self.process_service.complete_process(db, current, outputs=outputs)

    async def build_proposal(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
    ) -> dict:
        required = infer_required_work_functions(goal)
        review_process = await self.process_service.get_current(
            db, goal.id, "agent_definition_review"
        )
        review_ids = self._review_ids(review_process.outputs if review_process else {})
        reviews = [
            review
            for review in await self.review_service.list_reviews(db, goal.id)
            if review.id in review_ids and review.agent_id is not None
        ]
        agent_ids = {review.agent_id for review in reviews if review.agent_id is not None}
        agents = await self._agents_by_id(db, agent_ids)
        displayed = {
            work_function: await self.roster_mapper.rank_agents(
                db, goal.project_id, work_function
            )
            for work_function in required
        }
        eligible = {
            review.agent_id
            for review in reviews
            if review.agent_id in agents
            and review.agent_id is not None
            and review.approved_for_work_functions
        }

        candidate_agents: dict[str, list[str]] = {}
        suggestions: list[dict[str, str]] = []
        canonical: dict[str, list[uuid.UUID]] = {}
        weak_fits: list[dict[str, str]] = []
        for work_function in required:
            approved = {
                review.agent_id
                for review in reviews
                if review.agent_id in eligible
                and work_function in (review.approved_for_work_functions or [])
            }
            strong = {
                agent_id
                for agent_id in approved
                if self.roster_mapper.score_agent_definition(agents[agent_id], work_function).score
                >= WEAK_FIT_THRESHOLD
            }
            weak = approved - strong
            candidate_agents[work_function] = [
                str(fit.agent_id) for fit in displayed[work_function] if fit.agent_id in strong
            ]
            weak_fits.extend(
                {"work_function": work_function, "agent_id": str(agent_id)}
                for agent_id in sorted(
                    weak,
                    key=lambda agent_id: self._canonical_sort_key(agents[agent_id], work_function),
                )
            )
            suggestions.extend(
                {"work_function": work_function, "agent_id": str(fit.agent_id)}
                for fit in displayed[work_function]
                if fit.agent_id not in strong
            )
            canonical[work_function] = sorted(
                strong,
                key=lambda agent_id: self._canonical_sort_key(agents[agent_id], work_function),
            )

        role_to_agent: dict[str, str] = {}
        producer_ids: set[uuid.UUID] = set()
        independent_functions = self._independent_functions(goal)
        producer_functions = [name for name in required if name not in _VERIFIER_FUNCTIONS]
        for work_function in producer_functions:
            if canonical[work_function]:
                role_to_agent[work_function] = str(canonical[work_function][0])
                producer_ids.add(canonical[work_function][0])

        for work_function in (name for name in required if name in _VERIFIER_FUNCTIONS):
            candidates = canonical[work_function]
            safe_candidates = [
                agent_id
                for agent_id in candidates
                if self._independent_of(agent_id, producer_ids, goal)
            ]
            if safe_candidates or work_function in independent_functions:
                candidates = safe_candidates
            if candidates:
                role_to_agent[work_function] = str(candidates[0])

        hierarchy = self._hierarchy(goal, role_to_agent)
        independent_verification = self._independent_verification(
            independent_functions, role_to_agent
        )
        return {
            "source_agent_review_process_id": str(review_process.id) if review_process else None,
            "review_ids": sorted(str(review.id) for review in reviews),
            "weight": goal.weight,
            "compressed": goal.weight != "substantial",
            "required_work_functions": required,
            "candidate_agents": candidate_agents,
            "role_to_agent": role_to_agent,
            "hierarchy": hierarchy,
            "weak_fits": weak_fits,
            "responsibility_conflicts": self._responsibility_conflicts(role_to_agent),
            "independent_verification": independent_verification,
            "missing_work_functions": [
                work_function for work_function in required if work_function not in role_to_agent
            ],
            "suggestions": suggestions,
        }

    @staticmethod
    def _review_ids(outputs: dict) -> set[uuid.UUID]:
        ids: set[uuid.UUID] = set()
        for value in outputs.get("review_ids", []) if isinstance(outputs, dict) else []:
            try:
                ids.add(uuid.UUID(str(value)))
            except (TypeError, ValueError):
                continue
        return ids

    @staticmethod
    async def _agents_by_id(db: AsyncSession, agent_ids: set[uuid.UUID]) -> dict[uuid.UUID, Agent]:
        if not agent_ids:
            return {}
        result = await db.execute(select(Agent).where(Agent.id.in_(agent_ids), Agent.is_active.is_(True)))
        return {agent.id: agent for agent in result.scalars()}

    def _canonical_sort_key(self, agent: Agent, work_function: str) -> tuple[int, str, str]:
        fit = self.roster_mapper.score_agent_definition(agent, work_function)
        return (-fit.score, agent.name.lower(), str(agent.id))

    @staticmethod
    def _responsibility_conflicts(role_to_agent: dict[str, str]) -> list[dict[str, object]]:
        by_agent: dict[str, list[str]] = {}
        for work_function, agent_id in role_to_agent.items():
            by_agent.setdefault(agent_id, []).append(work_function)
        return [
            {"agent_id": agent_id, "work_functions": work_functions}
            for agent_id, work_functions in sorted(by_agent.items())
            if set(work_functions).intersection(_VERIFIER_FUNCTIONS)
            and set(work_functions).difference(_VERIFIER_FUNCTIONS)
        ]

    @staticmethod
    def _hierarchy(goal: OrchestrationGoal, role_to_agent: dict[str, str]) -> dict[str, object]:
        def role_ids(work_functions: tuple[str, ...]) -> list[str]:
            return list(dict.fromkeys(
                role_to_agent[work_function]
                for work_function in work_functions
                if work_function in role_to_agent
            ))

        return {
            "manager": TeamHierarchyProcess._manager(goal),
            "team_leads": [],
            "contributors": role_ids(("planning", "investigation", "implementation", "summarization")),
            "reviewers": role_ids(("review",)),
            "validators": role_ids(("validation",)),
            "specialists": [],
        }

    @staticmethod
    def _manager(goal: OrchestrationGoal) -> dict[str, str | None]:
        if goal.manager_agent_id is not None:
            return {"kind": "agent", "id": str(goal.manager_agent_id)}
        if goal.manager_user_id is not None:
            return {"kind": "human", "id": str(goal.manager_user_id)}
        return {"kind": "none", "id": None}

    @staticmethod
    def _independent_functions(goal: OrchestrationGoal) -> tuple[str, ...]:
        """Verifier functions that must be performed by an independent agent."""
        if requires_independent_verification(goal.success_criteria, goal.objective):
            return ("review", "validation")
        return ("validation",) if goal.weight == "substantial" else ()

    @staticmethod
    def _independent_of(agent_id, producer_ids, goal: OrchestrationGoal) -> bool:
        """A verifier is independent of every producer and of the goal's manager agent."""
        return (str(agent_id) not in {str(item) for item in producer_ids}
                and (goal.manager_agent_id is None or str(agent_id) != str(goal.manager_agent_id)))

    @staticmethod
    def _gates(*, missing: bool, possible: bool, documented_gaps: bool,
               agent_definitions_reviewed: bool) -> dict[str, bool]:
        """Accepted-hierarchy completion gates (shared by review and approved recovery)."""
        return {
            "team_structure_reviewed": True,
            "required_work_functions_mapped": not missing and not documented_gaps,
            "agent_definitions_reviewed": agent_definitions_reviewed,
            "missing_capabilities_handled": not missing or documented_gaps,
            "independent_verification_possible_or_overridden": possible or documented_gaps,
        }

    @staticmethod
    def _independent_verification(
        required_functions: tuple[str, ...], role_to_agent: dict[str, str]
    ) -> dict[str, object]:
        verifier_agent_ids = list(dict.fromkeys(
            role_to_agent[work_function]
            for work_function in required_functions
            if work_function in role_to_agent
        ))
        return {
            "required": bool(required_functions),
            "possible": all(
                work_function in role_to_agent
                for work_function in required_functions
            ),
            "verifier_agent_ids": verifier_agent_ids,
            "overridden_by_decision_id": None,
        }

    @staticmethod
    def _fingerprint(goal: OrchestrationGoal, proposal: dict) -> str:
        return trivial_hierarchy_fingerprint_from_projection(goal, proposal)

    @staticmethod
    def _gaps(proposal: dict, agent_definition_review_skipped: bool) -> list[str]:
        gaps = []
        if proposal["missing_work_functions"]:
            gaps.append("missing_work_functions")
        if proposal["weak_fits"]:
            gaps.append("weak_fits")
        if proposal["responsibility_conflicts"]:
            gaps.append("responsibility_conflicts")
        if proposal["hierarchy"]["manager"]["kind"] == "none":
            gaps.append("no_manager")
        if agent_definition_review_skipped:
            gaps.append("agent_definition_review_skipped")
        independent = proposal["independent_verification"]
        if independent["required"] and not independent["possible"]:
            gaps.append("no_independent_verifier")
        return gaps

    @staticmethod
    def _routing_gaps(proposal: dict, agent_definition_review_skipped: bool) -> list[str]:
        gaps = []
        if proposal["missing_work_functions"]:
            gaps.append("missing_work_functions")
        if proposal["hierarchy"]["manager"]["kind"] == "none":
            gaps.append("no_manager")
        if agent_definition_review_skipped:
            gaps.append("agent_definition_review_skipped")
        independent = proposal["independent_verification"]
        if independent["required"] and not independent["possible"]:
            gaps.append("no_independent_verifier")
        return gaps

    async def _agent_definition_review_was_skipped(
        self, db: AsyncSession, goal_id: uuid.UUID
    ) -> bool:
        current = await self.process_service.get_current(
            db, goal_id, "agent_definition_review"
        )
        return current is not None and current.status == "skipped"

    async def _approval_authority(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        *,
        gaps: list[str],
        force_human: bool = False,
    ) -> tuple[str, uuid.UUID | None]:
        if not gaps and not force_human and goal.manager_agent_id is not None:
            manager = await db.get(Agent, goal.manager_agent_id)
            if manager is not None and manager.is_active:
                return "manager", manager.id
        return "human", None

    async def _ask(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
        current: OrchestrationProcessRun,
        proposal: dict,
        *,
        gaps: list[str],
        routing_gaps: list[str],
        force_human: bool = False,
    ) -> OrchestrationAuthorityDecision:
        authority, authority_agent_id = await self._approval_authority(
            db, goal, gaps=routing_gaps, force_human=force_human
        )
        if authority == "human" and gaps:
            options = [
                {"key": "approve_with_documented_gaps", "proposal": proposal},
                {"key": "request_changes"},
            ]
        else:
            options = [
                {"key": "approve", "proposal": proposal},
                {"key": "request_changes"},
            ]
        if authority == "manager":
            options.append({"key": "escalate_to_human"})
        return await self.decision_service.create_pending(
            db,
            goal.id,
            decision_key=APPROVAL_DECISION_KEY,
            title="Approve team hierarchy",
            question="Approve the proposed team hierarchy and role assignments?",
            authority=authority,
            authority_agent_id=authority_agent_id,
            options=options,
            context=json.dumps({"gaps": gaps}, sort_keys=True),
            recommendation=(
                "approve_with_documented_gaps" if authority == "human" and gaps else "approve"
            ),
            consequences=(
                "Approval records the hierarchy used for delegation; requesting "
                "changes keeps the process parked."
            ),
            run_id=run.id,
            source_process_run_id=current.id,
        )

    async def _principal_is_current(
        self, db: AsyncSession, decision: OrchestrationAuthorityDecision
    ) -> bool:
        if decision.authority == "human":
            if decision.decided_by_user_id is None:
                return False
            if decision.decided_by_user_id.int == 0:
                return not settings.auth_enabled
            user = await db.get(User, decision.decided_by_user_id)
            return user is not None and user.is_active
        if decision.decided_by_agent_id is None:
            return False
        agent = await db.get(Agent, decision.decided_by_agent_id)
        return agent is not None and agent.is_active

    async def _cancel_pending_for_run(
        self,
        db: AsyncSession,
        goal_id: uuid.UUID,
        process_run_id: uuid.UUID,
    ) -> None:
        for decision in await self.decision_service.list_decisions(
            db, goal_id, status="pending"
        ):
            if decision.source_process_run_id == process_run_id:
                await self.decision_service.cancel_decision(
                    db,
                    decision,
                    reason="team hierarchy inputs changed",
                )

    async def _write_warnings(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
        current: OrchestrationProcessRun,
        proposal: dict,
    ) -> list:
        findings: list[tuple[str, str]] = []
        missing = proposal["missing_work_functions"]
        if missing:
            findings.append(
                (
                    "team_hierarchy_missing_work_functions",
                    "Required work functions have no reviewed eligible assignment: "
                    + ", ".join(missing)
                    + ".",
                )
            )
        if proposal["weak_fits"]:
            findings.append(
                (
                    "team_hierarchy_weak_role_fit",
                    "Some reviewed agents have only weak role fit for required work.",
                )
            )
        if proposal["responsibility_conflicts"]:
            findings.append(
                (
                    "team_hierarchy_unclear_boundaries",
                    "Producer and verifier responsibilities overlap in the proposed hierarchy.",
                )
            )
        independent = proposal["independent_verification"]
        if independent["required"] and not independent["possible"]:
            findings.append(
                (
                    "team_hierarchy_no_independent_verifier",
                    NO_INDEPENDENT_VERIFIER_MESSAGE,
                )
            )
        return [
            await self.warning_service.create_warning(
                db,
                goal.id,
                warning_type=warning_type,
                severity="warning",
                message=message,
                run_id=run.id,
                source_process_run_id=current.id,
            )
            for warning_type, message in findings
        ]

    async def _write_memory(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
        proposal: dict,
    ) -> None:
        if goal.weight == "trivial":
            body = (
                "Trivial goal: the human remains the implicit point of contact; "
                "no team hierarchy review was required."
            )
        else:
            required = ", ".join(proposal["required_work_functions"]) or "none"
            assignments = ", ".join(
                f"{work_function}: agent:{agent_id}"
                for work_function, agent_id in proposal["role_to_agent"].items()
            ) or "none"
            independent = proposal["independent_verification"]
            body = "\n".join(
                (
                    f"Required work functions: {required}",
                    f"Assignments: {assignments}",
                    "Independent verification: "
                    f"required={independent['required']}, possible={independent['possible']}",
                )
            )
        await self.memory_service.upsert_section(
            db,
            goal.project_id,
            goal.id,
            section_key=MEMORY_SECTION_KEY,
            title="Team hierarchy and role fit",
            body=body,
            summary=(
                f"Mapped {len(proposal['role_to_agent'])} of "
                f"{len(proposal['required_work_functions'])} required work functions."
            ),
            toc_order=MEMORY_TOC_ORDER,
            run_id=run.id,
            created_by="orchestrator:team_hierarchy",
        )

    async def _complete(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        current: OrchestrationProcessRun,
        proposal: dict,
        fingerprint: str,
        approval: OrchestrationAuthorityDecision | None,
        *,
        warnings: list | None = None,
        agent_definitions_reviewed: bool,
    ) -> None:
        outputs = json.loads(json.dumps(proposal))
        documented_gaps = (
            approval is not None
            and approval.selected_option == "approve_with_documented_gaps"
            and approval.authority == "human"
        )
        if documented_gaps:
            outputs["independent_verification"]["overridden_by_decision_id"] = str(
                approval.id
            )
        outputs.update(
            {
                "fingerprint": fingerprint,
                "compressed": goal.weight != "substantial",
                "approval": approval.selected_option if approval else None,
                "approval_decision_id": str(approval.id) if approval else None,
                "warning_ids": [str(warning.id) for warning in warnings or []],
                "gates": self._gates(
                    missing=bool(outputs["missing_work_functions"]),
                    possible=bool(outputs["independent_verification"]["possible"]),
                    documented_gaps=bool(documented_gaps),
                    agent_definitions_reviewed=agent_definitions_reviewed,
                ),
            }
        )
        await self._resolve_prior_skip_warnings(db, goal.id, current.id)
        await self.process_service.complete_process(db, current, outputs=outputs)

    @staticmethod
    def _persist_waiting_outputs(
        current: OrchestrationProcessRun,
        proposal: dict,
        fingerprint: str,
        warnings: list,
    ) -> None:
        if (
            current.outputs.get("fingerprint") == fingerprint
            and isinstance(current.outputs.get("proposal"), dict)
        ):
            return
        current.outputs = {
            "proposal": json.loads(json.dumps(proposal)),
            "fingerprint": fingerprint,
            "warning_ids": [str(warning.id) for warning in warnings],
            "change_requested": bool(current.outputs.get("change_requested")),
        }

    async def _resolve_prior_skip_warnings(
        self,
        db: AsyncSession,
        goal_id: uuid.UUID,
        current_process_id: uuid.UUID,
        *,
        all_types: bool = False,
        reason: str = "superseded by a successful team hierarchy review",
    ) -> None:
        """Resolve prior hierarchy rows' skip warnings (``all_types``: every active warning they raised)."""
        prior_ids = set(
            (
                await db.execute(
                    select(OrchestrationProcessRun.id).where(
                        OrchestrationProcessRun.goal_id == goal_id,
                        OrchestrationProcessRun.process_type == PROCESS_TYPE,
                        OrchestrationProcessRun.id != current_process_id,
                    )
                )
            ).scalars()
        )
        if not prior_ids:
            return
        for warning in await self.warning_service.list_warnings(
            db, goal_id, active_only=True
        ):
            if (
                warning.source_process_run_id in prior_ids
                and (all_types or warning.warning_type
                     in {"team_hierarchy_skipped", "team_hierarchy_not_reviewed"})
            ):
                await self.warning_service.resolve_warning(
                    db,
                    warning,
                    resolved_by="orchestrator:team_hierarchy",
                    reason=reason,
                )

    @staticmethod
    def _summary(status: str, questions_created: int = 0) -> dict:
        return {
            "process_type": PROCESS_TYPE,
            "status": status,
            "questions_created": questions_created,
        }


PLAN_AUTHOR_FUNCTIONS = {
    "roadmap": ("management", "planning"),
    "delivery": ("management", "planning"),
    "implementation": ("planning", "implementation"),
}


def plan_mode(goal: OrchestrationGoal, run: OrchestrationRun) -> str:
    return run.planning_mode_snapshot or goal.planning_mode


def accepted_role_map(team) -> dict[str, str]:
    """work_function -> agent id from either accepted team shape."""
    team = team if isinstance(team, dict) else {}
    roles = {k: str(v) for k, v in (team.get("role_to_agent") or {}).items()}
    for item in team.get("assignments") or []:
        if isinstance(item, dict) and item.get("work_function") and item.get("agent_ref"):
            roles.setdefault(item["work_function"], str(item["agent_ref"]))
    return roles


async def select_plan_author(
    db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun
) -> uuid.UUID | None:
    """The one compatible plan author, or None when no accepted team names candidates."""
    from fastapi import HTTPException
    from huddleroom.services.orchestration_agent_definition_review import (
        accepted_work_functions,
        pending_definition_repair,
    )

    mode = plan_mode(goal, run)
    if mode == "continuous_policy":
        raise HTTPException(
            409,
            "plan_not_applicable: continuous_policy goals run their accepted Continuous policy; "
            "no plan task is created",
        )
    root, team = goal, None
    while True:
        team = (root.orchestrator_context or {}).get("team") or None
        if not team:
            current = await OrchestrationProcessService().get_current(db, root.id, PROCESS_TYPE)
            if current is not None and current.status == "completed":
                team = current.outputs
        if team or root.parent_goal_id is None:
            break
        root = await db.get(OrchestrationGoal, root.parent_goal_id)
    roles = accepted_role_map(team)
    if not roles:
        return None
    # nearest ancestor (self first) with a reviewed eligibility map
    review_goal, eligibility = goal, None
    while True:
        eligibility = await accepted_work_functions(db, review_goal.id)
        if eligibility is not None or review_goal.parent_goal_id is None:
            break
        review_goal = await db.get(OrchestrationGoal, review_goal.parent_goal_id)
    functions = PLAN_AUTHOR_FUNCTIONS[mode]
    candidates: list[tuple[str, str]] = []
    for fn in functions:
        agent_id = str(goal.manager_agent_id) if fn == "management" and goal.manager_agent_id else roles.get(fn)
        if agent_id:
            candidates.append((fn, agent_id))
    for fn, agent_id in candidates:
        try:
            agent = await db.get(Agent, uuid.UUID(agent_id))
        except ValueError:
            continue
        # ponytail: absent = not reviewed for this goal -> trust the accepted team
        if agent and agent.is_active and (
            eligibility is None or agent_id not in eligibility or fn in eligibility[agent_id]
        ):
            return agent.id
    if not candidates:
        return None
    repair = "update the team hierarchy or the agent definition and rerun the agent definition review"
    for _fn, agent_id in candidates:
        decision = await pending_definition_repair(db, review_goal.id, agent_id)
        if decision is not None:
            repair = f"resolve agent definition decision {decision.id} (approve or edit)"
            break
    raise HTTPException(
        409,
        f"plan_author_unavailable: no accepted {' or '.join(functions)} owner is compatible "
        f"for {mode} planning; {repair}",
    )
