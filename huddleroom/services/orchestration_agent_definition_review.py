"""Pure deterministic assessment of an agent definition."""

# pylint: disable=too-many-locals

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import re
import uuid
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.config import settings
from huddleroom.models.agent import Agent
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGoal, OrchestrationRun
from huddleroom.models.orchestration_process import (
    OrchestrationAgentReview,
    OrchestrationAuthorityDecision,
    OrchestrationProcessRun,
    OrchestrationWarning,
)
from huddleroom.models.task import Task
from huddleroom.schemas.orchestration_recovery import EvidenceV1, _EligibilityV1
from huddleroom.services.orchestration_agent_definition_analyzer import (
    AgentDefinitionSemanticAnalyzer,
    redact_semantic_payload,
)
from huddleroom.services.orchestration_agent_review_service import (
    FINGERPRINT_SNAPSHOT_FIELDS,
    SNAPSHOT_FIELDS,
    OrchestrationAgentReviewService,
)
from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
from huddleroom.services.orchestration_llm_decision_adapter import _full_completion_error, _safe_completion_error
from huddleroom.services.orchestration_memory_service import OrchestrationMemoryService
from huddleroom.services.orchestration_process_service import (
    OrchestrationProcessService,
    effective_input_fingerprint,
    capture_input_fingerprint,
    may_initialize_capture,
    set_effective_input_fingerprint,
)
from huddleroom.services.orchestration_roster_mapper import OrchestrationRosterMapper, PROFILES
from huddleroom.services.orchestration_warning_service import OrchestrationWarningService


CRITICAL_VALIDATION_FUNCTIONS = {"review", "validation"}
WEAK_REASONING_EFFORTS = {"none", "minimal", "low"}
PROCESS_TYPE = "agent_definition_review"
PROCESS_VERSION = 3
# Bump when assess_agent_definition() rules change; old completed reviews are refreshed without the analyzer.
DETERMINISTIC_POLICY = 2
DECISION_KEY_PREFIX = "agent_definition_review:proposal:"
MEMORY_SECTION_KEY = "agent_definition_review"
MEMORY_TOC_ORDER = 30
SKIP_WARNING_TYPE = "agent_definition_review_not_performed"
SKIP_WARNING_MESSAGE = (
    "Agent definitions were not reviewed before delegation. "
    "Some agents may be misconfigured for their assigned work functions."
)
ARTIFACT_ACTION_RE = re.compile(
    r"\b(?:implement(?:s|ing)?|edit(?:s|ing)?(?:\s+(?:files?|documents?))?|"
    r"(?:produce|create|write|build)(?:s|ing)?\s+(?:code|files?|documents?|work\s+artifacts?|artifacts?|deliverables?))\b",
    re.IGNORECASE,
)
DOCUMENT_ARTIFACT_ACTION_RE = re.compile(
    r"\b(?:"
    r"(?:edit(?:s|ing)?|build(?:s|ing)?|(?:produc|creat|writ)(?:e|es|ing)?)\s+"
    r"(?:(?:markdown|text)\s+)?(?:files?|documents?|documentation)"
    r"|(?:summariz|writ)(?:e|es|ing)?\s+summaries?\s+(?:to|into)\s+files?"
    r")\b",
    re.IGNORECASE,
)
NEGATION_RE = re.compile(
    r"\b(?:(?:do|does|must|should|will|can)\s+not|cannot|can't|don't|doesn't|mustn't|shouldn't|won't)\b"
    r"|\bnever\b|\bwithout\b",
    re.IGNORECASE,
)
CLAUSE_SPLIT_RE = re.compile(r"[.!?;\n]+|,?\s+but\s+", re.IGNORECASE)
SELF_SUBJECT_RE = re.compile(
    r"(?:\b(?:you|the\s+agent|the\s+manager|manager)\s+"
    r"(?:(?:must|should|will|shall|directly)\s+)?|\byour\s+role\s+is\s+to\s+)$",
    re.IGNORECASE,
)
SELF_REFERENCE_RE = re.compile(r"\byourself\b", re.IGNORECASE)
NEGATED_SELF_REFERENCE_RE = re.compile(
    r"\b(?:not|never|without)\b[^,;.!?]*\byourself\b",
    re.IGNORECASE,
)
COORDINATED_IMPERATIVE_RE = re.compile(
    r"^\s*(?:manage|coordinate|own|lead|decide|prioritize)\b.*\b(?:and|then)\s+$",
    re.IGNORECASE,
)


def _warning(warning_type: str, message: str) -> dict[str, str]:
    return {"warning_type": warning_type, "severity": "warning", "message": message}


def _implied_work_function(value: str | None) -> str | None:
    mapper = OrchestrationRosterMapper()
    tokens = mapper._tokens(value)  # pylint: disable=protected-access
    matches = {
        name
        for name, profile in PROFILES.items()
        if any(
            mapper._matches_tokens(mapper._normalize_token(term), tokens)  # pylint: disable=protected-access
            for term in (profile.name, *profile.role_terms)
        )
    }
    return next(iter(matches)) if len(matches) == 1 else None


def _explicit_artifact_work(agent: Agent, action_re: re.Pattern[str] = ARTIFACT_ACTION_RE) -> bool:
    text = "\n".join(value for value in (agent.description, agent.system_prompt) if value)
    for clause in CLAUSE_SPLIT_RE.split(text):
        for match in action_re.finditer(clause):
            prefix = clause[: match.start()]
            if NEGATION_RE.search(prefix):
                continue
            suffix = clause[match.end() :]
            positive_self_reference = (
                SELF_REFERENCE_RE.search(suffix)
                and not NEGATED_SELF_REFERENCE_RE.search(suffix)
            )
            if (
                not prefix.strip()
                or SELF_SUBJECT_RE.search(prefix)
                or positive_self_reference
                or COORDINATED_IMPERATIVE_RE.search(prefix)
            ):
                return True
    return False


def assess_agent_definition(
    agent: Agent,
    proposed_work_functions: list[str],
    *,
    inactive_assignee: bool = False,
) -> dict[str, Any]:
    """Assess explicit definition facts and return persistence-ready fields."""
    proposed = list(proposed_work_functions)
    normalized = {
        work_function: OrchestrationRosterMapper._normalize_token(work_function)  # pylint: disable=protected-access
        for work_function in proposed
    }
    config = agent.config if isinstance(agent.config, dict) else {}
    warnings: list[dict[str, str]] = []
    disqualified: set[str] = set()
    risks: list[str] = []
    recommended_changes: list[str] = []

    name_function = _implied_work_function(agent.name)
    role_function = _implied_work_function(agent.role)
    if name_function and role_function and name_function != role_function:
        warnings.append(
            _warning(
                "agent_review_role_name_mismatch",
                f"Agent name indicates {name_function}, while its role indicates {role_function}.",
            )
        )
        disqualified.update({name_function, role_function})
        recommended_changes.append("Align the agent name and role with its intended work functions.")

    if not (agent.description or "").strip() and not (agent.system_prompt or "").strip():
        warnings.append(
            _warning(
                "agent_review_vague_definition",
                f"Agent '{agent.name}' has no description or system prompt, so its work cannot be assessed. Add a concrete description or system prompt.",
            )
        )
        disqualified.update(normalized.values())
        recommended_changes.append("Add a concrete description or system prompt.")

    critical_proposed = CRITICAL_VALIDATION_FUNCTIONS.intersection(normalized.values())
    temperature = config.get("temperature")
    weak_temperature = (
        agent.adapter_type == "api"
        and isinstance(temperature, (int, float))
        and not isinstance(temperature, bool)
        and temperature > 0.3
    )
    reasoning_effort = config.get("reasoning_effort")
    weak_reasoning = (
        isinstance(reasoning_effort, str)
        and reasoning_effort.strip().lower() in WEAK_REASONING_EFFORTS
    )
    if critical_proposed and (weak_temperature or weak_reasoning):
        warnings.append(
            _warning(
                "agent_review_weak_validation_config",
                "Review or validation work has an explicitly weak deterministic configuration.",
            )
        )
        disqualified.update(CRITICAL_VALIDATION_FUNCTIONS)
        recommended_changes.append(
            "Use API temperature at or below 0.3 and reasoning_effort above low for review work."
        )

    if config.get("allow_global_scope") is True:
        warnings.append(
            _warning(
                "agent_review_unsafe_permissions",
                "Agent explicitly has global scope.",
            )
        )
        disqualified.update(normalized.values())
        recommended_changes.append(
            "Remove allow_global_scope from the agent config, or confirm global scope is intended."
        )

    if inactive_assignee and not bool(agent.is_active):
        warnings.append(
            _warning(
                "agent_review_inactive_assignee",
                "Inactive agent remains assigned to the current run.",
            )
        )
        disqualified.update(normalized.values())
        recommended_changes.append("Reassign the work to an active agent.")

    if "management" in normalized.values() and _explicit_artifact_work(agent):
        warnings.append(
            _warning(
                "agent_review_manager_overreach",
                "Management instructions explicitly require implementation or artifact production.",
            )
        )
        disqualified.add("management")
        recommended_changes.append("Keep management instructions focused on coordination and decisions.")

    api_implementation = agent.adapter_type == "api" and "implementation" in normalized.values()
    api_document_work = (
        agent.adapter_type == "api"
        and "summarization" in normalized.values()
        and _explicit_artifact_work(agent, DOCUMENT_ARTIFACT_ACTION_RE)
    )
    if api_implementation or api_document_work:
        warnings.append(
            _warning(
                "agent_review_api_workspace_required",
                "API agents do not have a workspace toolchain for implementation or file-producing document work.",
            )
        )
        if api_implementation:
            disqualified.add("implementation")
        if api_document_work:
            disqualified.add("summarization")
        recommended_changes.append(
            "Use a CLI adapter for implementation or file-producing document work."
        )

    for key in ("temperature", "reasoning_effort", "tools"):
        if key not in config:
            risks.append(f"config.{key} is not declared; this optional setting is unreviewable.")

    strengths = []
    definition = ((agent.description or "").strip() or (agent.system_prompt or "").strip())
    if definition:
        strengths.append(f"Definition excerpt: {definition[:120]}")
    if agent.capabilities:
        strengths.append(f"Declared capabilities: {', '.join(map(str, agent.capabilities))}")
    if "tools" in config:
        strengths.append(f"Declared config.tools (descriptive only): {config['tools']!r}")

    approved = [
        work_function
        for work_function in proposed
        if normalized[work_function] not in disqualified
    ]
    provider_model = f"{agent.provider}/{agent.model}"
    fit_summary = (
        f"Deterministic definition review for {provider_model}: "
        f"{len(approved)} of {len(proposed)} proposed work functions eligible."
    )
    return {
        "fit_summary": fit_summary,
        "proposed_work_functions": proposed,
        "strengths": strengths,
        "risks": risks,
        "recommended_changes": recommended_changes,
        "approved_for_work_functions": approved,
        "warnings": warnings,
    }


RECOVERY_TRIGGER_REASON = "approved_recovery"


class RecoveryDefinitionAcceptanceV1(BaseModel):
    """Typed input of the recovery-sourced agent_definition_review snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    call_id: str = Field(min_length=1, max_length=100)
    canonical_request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    eligibility: dict[uuid.UUID, list[str]] = Field(min_length=1, max_length=128)
    changed_agent_ids: list[uuid.UUID] = Field(default_factory=list, max_length=128)

    @field_validator("eligibility")
    @classmethod
    def _functions(cls, value: dict[uuid.UUID, list[str]]) -> dict[uuid.UUID, list[str]]:
        for functions in value.values():
            _EligibilityV1._functions(functions)  # pylint: disable=protected-access
            if not functions:
                raise ValueError("eligibility requires at least one work function")
        return value


def recovery_eligibility_assessment(agent: Any, work_functions: list[str]) -> dict[str, Any]:
    """Deterministic assessment of an exact post-state agent for its approved functions.

    Recovery prose cannot override a hard disqualifier: every approved function must
    survive ``assess_agent_definition`` unchanged, and the agent must be active.
    """
    if not bool(getattr(agent, "is_active", False)):
        raise ValueError("inactive_or_unknown_agent")
    assessment = assess_agent_definition(agent, list(work_functions))
    if assessment["approved_for_work_functions"] != list(work_functions):
        raise ValueError("definition_disqualified")
    return assessment


def coverage_fingerprint_from_projection(
    goal: OrchestrationGoal,
    targets: list[tuple[Agent, list[str], bool]],
    *,
    fields: tuple[str, ...] = FINGERPRINT_SNAPSHOT_FIELDS,
) -> str:
    """Hash the side-effect-free coverage projection shared by roster previews."""
    payload = {
        "weight": goal.weight,
        "objective": goal.objective,
        "success_criteria": copy.deepcopy(goal.success_criteria or []),
        "constraints": copy.deepcopy(goal.constraints or {}),
        "authority_model": goal.authority_model,
        "selected_manager": (
            str(goal.manager_agent_id or goal.manager_user_id)
            if goal.manager_agent_id or goal.manager_user_id else None
        ),
        "targets": [
            {"agent_id": str(agent.id), "proposed_work_functions": proposed,
             "definition_snapshot": {field: copy.deepcopy(getattr(agent, field)) for field in fields}}
            for agent, proposed, _inactive in targets
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def coverage_targets_from_projection(
    goal: OrchestrationGoal,
    assignments: list[tuple[Task, Agent, str]],
    *,
    active_agents: list[Agent] = (),
    manager: Agent | None = None,
    roster_mapper: OrchestrationRosterMapper | None = None,
    required_functions: Sequence[str] = (),
) -> list[tuple[Agent, list[str], bool]]:
    """Build the coverage target payload without database access."""
    if goal.weight == "trivial":
        return []
    proposed_by_agent: dict[Any, set[str]] = {}
    assigned_ids = set()
    agents = {agent.id: agent for _task, agent, _function in assignments}
    for _task, agent, work_function in assignments:
        assigned_ids.add(agent.id)
        if work_function and (goal.weight == "standard" or not bool(agent.is_active)):
            proposed_by_agent.setdefault(agent.id, set()).add(work_function)
    if goal.weight == "standard":
        if goal.authority_model == "agent_manager" and manager is not None:
            agents[manager.id] = manager
            proposed_by_agent.setdefault(manager.id, set()).add("management")
        if required_functions:
            if roster_mapper is None:
                roster_mapper = OrchestrationRosterMapper()
            # Candidates are unassigned non-manager agents, only for required functions no assignee/manager already covers.
            manager_added = goal.authority_model == "agent_manager" and manager is not None
            covered = {work_function for _task, _agent, work_function in assignments if work_function}
            reserved_ids = set(assigned_ids)
            if manager_added:
                covered.add("management")
                reserved_ids.add(manager.id)
            open_fns = [f for f in required_functions if f not in covered]
            for agent in active_agents:
                if agent.id in reserved_ids:
                    continue
                fns = [
                    f for f in open_fns
                    if not roster_mapper.score_agent_definition(agent, f).weak
                ]
                if fns:
                    agents[agent.id] = agent
                    proposed_by_agent.setdefault(agent.id, set()).update(fns)
    else:
        if roster_mapper is None:
            roster_mapper = OrchestrationRosterMapper()
        for agent in active_agents:
            agents[agent.id] = agent
            proposed_by_agent.setdefault(agent.id, set()).update(
                work_function
                for work_function in PROFILES
                if not roster_mapper.score_agent_definition(agent, work_function).weak
            )
    return [
        (
            agent,
            sorted(proposed_by_agent.get(agent.id, set())),
            agent.id in assigned_ids and not bool(agent.is_active),
        )
        for agent in sorted(agents.values(), key=lambda value: str(value.id))
    ]


class AgentDefinitionReviewProcess:
    """Run the Phase 7 definition review synchronously."""

    def __init__(self, analyzer: AgentDefinitionSemanticAnalyzer | None = None) -> None:
        self.analyzer = analyzer or AgentDefinitionSemanticAnalyzer()
        self.process_service = OrchestrationProcessService()
        self.review_service = OrchestrationAgentReviewService()
        self.warning_service = OrchestrationWarningService()
        self.memory_service = OrchestrationMemoryService()
        self.decision_service = OrchestrationAuthorityDecisionService()
        self.roster_mapper = OrchestrationRosterMapper()

    @staticmethod
    def _goal_snapshot(goal: OrchestrationGoal) -> dict[str, Any]:
        return redact_semantic_payload({
            "objective": goal.objective,
            "success_criteria": copy.deepcopy(goal.success_criteria or []),
            "constraints": copy.deepcopy(goal.constraints or {}),
            "weight": goal.weight,
        })

    async def _proposal_decisions(
        self, db: AsyncSession, goal_id
    ) -> list[OrchestrationAuthorityDecision]:
        return [
            decision for decision in await self.decision_service.list_decisions(db, goal_id)
            if decision.decision_key.startswith(DECISION_KEY_PREFIX)
        ]

    @staticmethod
    def _decision_context(decision: OrchestrationAuthorityDecision) -> dict[str, Any]:
        payload = json.loads(decision.context or "{}")
        if not isinstance(payload, dict):
            raise ValueError("agent-definition decision context must be an object")
        return payload

    def _targets_before_answers(self, targets, current, decisions) -> list:
        """Targets with this process's answered approve/edit text reverted to the
        proposal-time originals (the state the stored input fingerprint saw)."""
        originals = {}
        for decision in decisions:
            if (
                decision.source_process_run_id == current.id
                and decision.status == "answered"
                and decision.selected_option in {"approve", "edit"}
            ):
                context = self._decision_context(decision)
                originals[context.get("agent_id")] = (
                    context.get("original_description"), context.get("original_persona"),
                )
        if not originals:
            return targets
        reverted = []
        for agent, proposed, inactive in targets:
            if str(agent.id) in originals:
                description, persona = originals[str(agent.id)]
                agent = SimpleNamespace(**{
                    **{field: getattr(agent, field) for field in SNAPSHOT_FIELDS},
                    "id": agent.id, "is_active": agent.is_active,
                    "description": description, "system_prompt": persona,
                })
            reverted.append((agent, proposed, inactive))
        return reverted

    async def _apply_answered_proposals(
        self,
        db: AsyncSession,
        run: OrchestrationRun,
        current: OrchestrationProcessRun,
        decisions: list[OrchestrationAuthorityDecision],
    ) -> bool:
        processed = set(current.outputs.get("processed_decision_ids", []))
        changed = False
        edits = {}
        invalid_edits = []
        for decision in decisions:
            if decision.selected_option != "edit" or str(decision.id) in processed:
                continue
            try:
                edited = json.loads(decision.reason or "{}")
            except (json.JSONDecodeError, TypeError):
                edited = None
            if not isinstance(edited, dict) or any(
                not isinstance(edited.get(key), str) or not edited[key].strip()
                for key in ("description", "persona")
            ):
                invalid_edits.append(decision)
                continue
            else:
                edits[decision.id] = edited
        if invalid_edits:
            for decision in invalid_edits:
                await self.decision_service.create_pending(
                    db, decision.goal_id, decision_key=decision.decision_key,
                    title=decision.title, question=decision.question, authority=decision.authority,
                    options=decision.options, context=decision.context,
                    recommendation=decision.recommendation, consequences=decision.consequences,
                    run_id=decision.run_id, source_process_run_id=decision.source_process_run_id,
                )
                processed.add(str(decision.id))
            current.outputs = {**current.outputs, "processed_decision_ids": sorted(processed)}
            await db.flush()
            return False
        from huddleroom.services.orchestration_roster_mutation import ValidatedRosterDelta, _canonical_agent

        targets = {}
        for decision in decisions:
            if decision.status != "answered" or str(decision.id) in processed:
                continue
            context = self._decision_context(decision)
            agent = await db.get(Agent, uuid.UUID(context["agent_id"]), populate_existing=True)
            if agent is None:
                raise ValueError(f"agent {context['agent_id']} no longer exists")
            if decision.selected_option in {"approve", "edit"}:
                description = context["proposed_description"]
                persona = context["proposed_persona"]
                if decision.selected_option == "edit":
                    edited = edits[decision.id]
                    description = edited.get("description")
                    persona = edited.get("persona")
                targets[decision.id] = (agent, description.strip(), persona.strip())
        updates = {
            agent.id: dataclasses.replace(_canonical_agent(agent), description=description, system_prompt=persona)
            for agent, description, persona in targets.values()
            if (agent.description, agent.system_prompt) != (description, persona)
        }
        if updates:
            # Global agent rows are written roster-first; under a goal lock this
            # hands off (the answered decisions are the durable reservation).
            from huddleroom.services.agent_service import hand_off_roster_delta

            await hand_off_roster_delta(
                db, ValidatedRosterDelta(updates=tuple(updates.values())),
                goal_id=current.goal_id, run_id=run.id, process_id=current.id,
                decision_ids=sorted(targets, key=str),
            )
        for decision in decisions:
            if decision.status != "answered" or str(decision.id) in processed:
                continue
            if decision.id in targets:
                agent = targets[decision.id][0]
                changed = True
                # The human just provided a definition, so the "both blank"
                # warning this run raised for the agent is provably obsolete.
                for warning in await self.warning_service.list_warnings(
                    db, decision.goal_id, active_only=True
                ):
                    if (
                        warning.related_agent_id == agent.id
                        and warning.warning_type == "agent_review_vague_definition"
                    ):
                        await self.warning_service.resolve_warning(
                            db,
                            warning,
                            resolved_by="orchestrator:agent_definition_review",
                            reason="definition provided via approved proposal",
                        )
            processed.add(str(decision.id))
        current.outputs = {**current.outputs, "processed_decision_ids": sorted(processed)}
        await db.flush()
        return changed

    def _retry_checkpoint(
        self,
        current: OrchestrationProcessRun,
        goal: OrchestrationGoal,
        targets: list[tuple[Agent, list[str], bool]],
        loads: dict,
        coverage_fingerprint: str,
    ) -> dict[str, Any]:
        return redact_semantic_payload({
            "version": 1,
            "kind": PROCESS_TYPE,
            "fingerprint": (current.outputs or {}).get("fingerprint"),
            "coverage_fingerprint": coverage_fingerprint,
            "targets": [
                {
                    "agent_id": str(agent.id),
                    "agent_snapshot": {
                        field: copy.deepcopy(getattr(agent, field)) for field in SNAPSHOT_FIELDS
                    },
                    "goal_snapshot": self._goal_snapshot(goal),
                    "candidate_work_functions": list(proposed),
                    "deterministic_assessment": copy.deepcopy(
                        assess_agent_definition(agent, proposed, inactive_assignee=inactive)
                    ),
                    "load_snapshot": {
                        "active_sessions": loads.get(agent.id).active_sessions if loads.get(agent.id) else 0,
                        "active_tasks": loads.get(agent.id).active_tasks if loads.get(agent.id) else 0,
                        "outcome_hints": loads.get(agent.id).outcome_hint_count if loads.get(agent.id) else 0,
                    },
                }
                for agent, proposed, inactive in targets
            ],
            "cursor": 0,
            "completed": {},
            "request": None,
            "error": None,
            "model": settings.orchestration_model,
        })

    def _semantic_payload(self, semantic, target: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "status": semantic.status,
            "problems": list(semantic.problems),
            "reason": semantic.reason,
            "approved_work_functions": list(semantic.approved_work_functions),
        }
        if semantic.status == "improvement_proposed":
            payload.update(
                proposed_description=semantic.proposed_description,
                proposed_persona=semantic.proposed_persona,
            )
        else:
            payload.update(proposed_description=None, proposed_persona=None)
        return payload

    async def _run_retry_checkpoint(
        self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun,
        current: OrchestrationProcessRun, checkpoint: dict[str, Any],
    ) -> dict | None:
        from huddleroom.models.project import Project
        _proj = await db.get(Project, goal.project_id)
        project_dict = {"name": _proj.name, "description": _proj.description} if _proj else None
        while checkpoint["cursor"] < len(checkpoint["targets"]):
            target = checkpoint["targets"][checkpoint["cursor"]]
            request = checkpoint["request"]
            if request is None:
                request = self.analyzer.build_request(
                    target["agent_snapshot"], target["goal_snapshot"], target["candidate_work_functions"],
                    project=project_dict,
                )
                request = {**request, "model": checkpoint["model"]}
            try:
                semantic = await self.analyzer.review_request(
                    request, target["candidate_work_functions"],
                    project_id=goal.project_id,
                )
            except Exception as exc:
                from huddleroom.services.orchestration_service import OrchestrationService

                safe_error = _safe_completion_error(exc)
                checkpoint["request"] = copy.deepcopy(getattr(exc, "request", request))
                checkpoint["error"] = safe_error
                evidence = {
                    "category": getattr(exc, "category", type(exc).__name__),
                    "request": getattr(exc, "request", request),
                    "raw_response": getattr(exc, "raw_response", None),
                    "error": getattr(exc, "full_error", _full_completion_error(exc)),
                }
                warning = await self.warning_service.create_warning(
                    db, goal.id, warning_type="agent_definition_review_analyzer_error",
                    severity="warning", message=f"Agent definition review failed: {safe_error}",
                    run_id=run.id, source_process_run_id=current.id,
                    related_agent_id=uuid.UUID(target["agent_id"]),
                )
                warning.message = f"Agent definition review failed: {safe_error}"
                checkpoint["warning_id"] = str(warning.id)
                current.outputs = {
                    **(current.outputs or {}), "_lm_retry": checkpoint,
                    "error": safe_error, "retryable": True, "model": settings.orchestration_model,
                    "semantic_error": redact_semantic_payload(evidence),
                }
                OrchestrationService._upsert_active_blocker(run, {
                    "kind": "agent_definition_review_analyzer_error",
                    "warning_id": str(warning.id),
                    "reason": f"Agent definition review failed: {safe_error}",
                })
                OrchestrationService._mark_run_blocked(goal, run)
                await db.flush()
                return {**self._summary(current.status), "retryable": True, "error": safe_error, "retry_failed": True}
            checkpoint["completed"][target["agent_id"]] = self._semantic_payload(semantic, target)
            await self._clear_recovered_failure(db, goal, run, checkpoint)
            checkpoint["cursor"] += 1
            checkpoint["request"] = None
            checkpoint["error"] = None
            current.outputs = {**(current.outputs or {}), "_lm_retry": checkpoint}
            await db.flush()
        return None

    async def _clear_recovered_failure(self, db, goal, run, checkpoint) -> None:
        warning_id = checkpoint.pop("warning_id", None)
        if not isinstance(warning_id, str):
            return
        for warning in await self.warning_service.list_warnings(db, goal.id, active_only=True):
            if str(warning.id) == warning_id:
                await self.warning_service.resolve_warning(
                    db, warning, resolved_by="orchestrator:agent_definition_review",
                    reason="agent definition review succeeded after analyzer failure",
                )
                break
        run.active_blockers = [item for item in run.active_blockers if not (
            isinstance(item, dict) and item.get("kind") == "agent_definition_review_analyzer_error"
            and item.get("warning_id") == warning_id
        )]

    async def retry_failed(
        self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun,
        current: OrchestrationProcessRun,
    ) -> dict:
        checkpoint = (current.outputs or {}).get("_lm_retry")
        if not isinstance(checkpoint, dict) or checkpoint.get("version") != 1 or checkpoint.get("kind") != PROCESS_TYPE:
            raise ValueError("agent definition review has no valid retry checkpoint")
        if checkpoint.get("fingerprint") != (current.outputs or {}).get("fingerprint"):
            raise ValueError("agent definition review retry checkpoint fingerprint mismatch")
        if not isinstance(checkpoint.get("cursor"), int) or not 0 <= checkpoint["cursor"] < len(checkpoint.get("targets", [])):
            raise ValueError("agent definition review retry checkpoint cursor is invalid")
        checkpoint["resumed"] = True
        result = await self._run_retry_checkpoint(db, goal, run, current, checkpoint)
        if result is not None:
            return result
        return await self._finalize_retry_checkpoint(db, goal, run, current, checkpoint)

    async def advance(
        self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun, *, manual: bool = False
    ) -> dict:
        current = await self.process_service.get_current(db, goal.id, PROCESS_TYPE)
        if current is None and not (run.baseline_authorized or manual):
            return self._summary("not_started")
        if current is not None and current.status == "skipped":
            await self.handle_skip(db, goal, current)
            return self._summary("skipped")
        if current is not None and isinstance((current.outputs or {}).get("_lm_retry"), dict):
            return {**self._summary(current.status), "retryable": True,
                    "error": current.outputs["_lm_retry"].get("error")}

        assignments, targets, fingerprint = await self._review_inputs(db, goal, run)
        if current is not None:
            capture_input_fingerprint(
                current, fingerprint, PROCESS_VERSION,
                allow_initialize=may_initialize_capture(current),
            )
        # Coverage a completed row is judged against: its own covered ids
        # (empty when absent, as in the readiness gate), not the current targets.
        accepted_coverage = None
        if current is not None and current.status == "completed":
            accepted_ids = (current.outputs or {}).get("target_ids")
            accepted_coverage = await self._coverage_fingerprint_for(
                db, goal, assignments,
                {str(x) for x in accepted_ids} if isinstance(accepted_ids, list) else set(),
            )
        proposal_decisions = await self._proposal_decisions(db, goal.id)
        stored_fingerprint = (
            effective_input_fingerprint(current)
            if current is not None and isinstance(current.input_snapshot, dict)
            else None
        )
        recorded = (current.outputs or {}).get("fingerprint") if current is not None else None
        if stored_fingerprint is None and current is not None and current.status == "waiting_decision" and isinstance(recorded, str):
            # Recorded reviewed input; no captured original is claimed.
            stored_fingerprint = recorded
        if (
            current is not None
            and current.status == "waiting_decision"
            and stored_fingerprint != fingerprint
            and stored_fingerprint == self._fingerprint(
                goal, assignments, self._targets_before_answers(targets, current, proposal_decisions)
            )
        ):
            # The only drift is this process's own approved text, applied
            # roster-first after the goal lock was released: re-stamp.
            set_effective_input_fingerprint(current, fingerprint, PROCESS_VERSION, "approved_definitions")
            stored_fingerprint = fingerprint
            await db.flush()
        if (
            current is not None
            and current.status == "waiting_decision"
            and stored_fingerprint != fingerprint
        ):
            # Stale inputs while a decision is pending: never cancel the
            # pending decision or restart -- park stays parked, a one-time
            # suggestion is raised instead (item 3).
            if run.phase in ("baseline", "ready"):
                await self.warning_service.suggest_stale_inputs(
                    db,
                    goal.id,
                    process_type=PROCESS_TYPE,
                    process_run_id=current.id,
                    step_label="agent definition review",
                    run_id=run.id,
                )
            await self.process_service.park_process(db, current)
            return self._summary("waiting_decision")
        current_decisions = [
            decision
            for decision in proposal_decisions
            if current is not None and decision.source_process_run_id == current.id
        ]
        if any(decision.status == "pending" for decision in current_decisions):
            await self.process_service.park_process(db, current)
            return self._summary("waiting_decision")

        processed = set((current.outputs or {}).get("processed_decision_ids", [])) if current else set()
        answered = [
            decision
            for decision in current_decisions
            if decision.status == "answered" and str(decision.id) not in processed
        ]
        if current is not None and answered:
            proposal_changed = await self._apply_answered_proposals(db, run, current, answered)
            if not proposal_changed:
                replacement_decisions = await self._proposal_decisions(db, goal.id)
                if any(
                    decision.status == "pending" and decision.source_process_run_id == current.id
                    for decision in replacement_decisions
                ):
                    await self.process_service.park_process(db, current)
                    return self._summary("waiting_decision")
            if current.status == "waiting_decision":
                await self.process_service.resume_process(db, current)
            assignments, refreshed_targets, refreshed_fingerprint = await self._review_inputs(
                db, goal, run
            )
            refreshed_ids = (current.outputs or {}).get("target_ids")
            if not isinstance(refreshed_ids, list):
                refreshed_ids = [str(agent.id) for agent, _, _ in refreshed_targets]
            refreshed_coverage = await self._coverage_fingerprint_for(
                db, goal, assignments, {str(x) for x in refreshed_ids}
            )
            current.outputs = {
                **(current.outputs or {}),
                "fingerprint": refreshed_fingerprint,
                "coverage_fingerprint": refreshed_coverage,
            }
            await db.flush()
            return await self._complete_persisted_review(db, goal, run, current)
        if (
            current is not None
            and current.status == "completed"
            and (current.outputs or {}).get("semantic_assessments")
            and (current.outputs or {}).get("deterministic_policy") != DETERMINISTIC_POLICY
        ):
            await self._refresh_deterministic_eligibility(db, goal, run, current, proposal_decisions)
        stored_cov = current.outputs.get("coverage_fingerprint") if current is not None else None
        legacy_coverage_restamp = (
            current is not None
            and current.status == "completed"
            and current.process_version == PROCESS_VERSION
            and stored_cov != accepted_coverage
            and (
                (stored_cov is None and current.outputs.get("fingerprint") == fingerprint)
                # Equality proves the full reviewed targets are unchanged; the
                # fingerprint check on the fast path still decides freshness.
                or stored_cov == self._coverage_fingerprint(goal, targets)
            )
        )
        if legacy_coverage_restamp:
            # Legacy row with no coverage fingerprint yet (backfill), or one
            # stored with the old full-target formula that the current state
            # still matches exactly: silently re-stamp, no warning/rerun.
            # Anything else stays stale. Falls through to the normal fast
            # path below, which will now match.
            current.outputs = {
                **(current.outputs or {}),
                "coverage_fingerprint": accepted_coverage,
            }
            await db.flush()
        elif (
            current is not None and current.status == "completed" and current.process_version != PROCESS_VERSION
        ):
            # A PROCESS_VERSION bump absorbs a fingerprint-*formula* change
            # (e.g. dropping provider/model from the hash). It must NOT also
            # silently absorb real drift that happens to land on the same
            # deploy. Recompute what the OLD formula (full SNAPSHOT_FIELDS)
            # would have produced against CURRENT state -- only if that
            # matches what's stored is this a pure formula change.
            old_formula_fingerprint = self._fingerprint(goal, assignments, targets, fields=SNAPSHOT_FIELDS)
            old_formula_coverage = self._coverage_fingerprint(goal, targets, fields=SNAPSHOT_FIELDS)
            stored_ids = (current.outputs or {}).get("target_ids")
            if not isinstance(stored_ids, list):
                stored_ids = []
            old_canonical_coverage = await self._coverage_fingerprint_for(
                db, goal, assignments, {str(x) for x in stored_ids}, fields=SNAPSHOT_FIELDS
            )
            stored_fp = current.outputs.get("fingerprint")
            pure_version_bump = stored_fp == old_formula_fingerprint and (
                stored_cov is None or stored_cov in (old_formula_coverage, old_canonical_coverage)
            )
            current.process_version = PROCESS_VERSION
            current.outputs = {
                **(current.outputs or {}),
                "process_version": PROCESS_VERSION,
            }
            if pure_version_bump:
                set_effective_input_fingerprint(current, fingerprint, PROCESS_VERSION, "formula_upgrade")
                current.outputs = {
                    **(current.outputs or {}),
                    "fingerprint": fingerprint,
                    "coverage_fingerprint": accepted_coverage,
                }
            # else: genuine drift co-occurring with the version bump -- the
            # version is bumped now so it isn't rechecked every tick, but the
            # stored fingerprint/coverage_fingerprint are left stale so the
            # normal stale-inputs path below fires (suggestion in baseline
            # phase / no rerun post-baseline unless user-initiated).
            await db.flush()
        if (
            current is not None
            and current.status == "completed"
            and current.process_version == PROCESS_VERSION
            and current.outputs.get("fingerprint") == fingerprint
            and current.outputs.get("coverage_fingerprint") == accepted_coverage
        ):
            return self._summary("completed")

        stored_fingerprint = (
            effective_input_fingerprint(current)
            if current is not None and isinstance(current.input_snapshot, dict)
            else None
        )
        running_inputs_changed = (
            current is not None
            and current.status == "running"
            and bool(stored_fingerprint)
            and stored_fingerprint != fingerprint
            and (manual or run.phase == "baseline")
        )
        if running_inputs_changed:
            superseded = current
            superseded.superseded_by_id = superseded.id
            await db.flush()
            current = await self.process_service.start_process(
                db,
                goal.id,
                process_type=PROCESS_TYPE,
                trigger_reason="auto-rerun: running review inputs changed",
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
                # auto-rerun (item 3) -- the completed review stands until a
                # human approves a rerun.
                if run.phase in ("baseline", "ready"):
                    await self.warning_service.suggest_stale_inputs(
                        db,
                        goal.id,
                        process_type=PROCESS_TYPE,
                        process_run_id=current.id,
                        step_label="agent definition review",
                        run_id=run.id,
                    )
                return self._summary("completed")
            # Post-baseline: the tick never reruns; only a user-initiated
            # (manual) rerun restarts the review. Execution-time task assignments move _fingerprint (task ids) but
            # not coverage_fingerprint; re-stamp the fingerprint silently.
            if (
                accepted_coverage is not None
                and current.outputs.get("coverage_fingerprint") == accepted_coverage
            ):
                set_effective_input_fingerprint(current, fingerprint, PROCESS_VERSION, "covered_workload")
                current.outputs = {**(current.outputs or {}), "fingerprint": fingerprint}
                await db.flush()
                return self._summary("completed")
            if not manual:
                # Gate only applies until Start; later it stays silent.
                if run.phase == "ready":
                    await self.warning_service.suggest_stale_inputs(
                        db,
                        goal.id,
                        process_type=PROCESS_TYPE,
                        process_run_id=current.id,
                        step_label="agent definition review",
                        run_id=run.id,
                    )
                return self._summary("completed")
            current = await self.process_service.start_process(
                db,
                goal.id,
                process_type=PROCESS_TYPE,
                trigger_reason="auto-rerun: review inputs changed",
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
                trigger_reason="tick: no agent definition review on record",
                run_id=run.id,
                input_snapshot={"fingerprint": fingerprint},
                process_version=PROCESS_VERSION,
            )
            # Bug #42: concurrent force-start can return a terminal row; short-circuit if not active
            if current.status not in ("running", "waiting_decision"):
                return self._summary(current.status)

        loads = await self.roster_mapper.loads_by_agent(db, goal.project_id) if targets else {}
        current.outputs = {**(current.outputs or {}), "fingerprint": fingerprint}
        review_coverage = await self._coverage_fingerprint_for(
            db, goal, assignments, {str(agent.id) for agent, _, _ in targets}
        )
        checkpoint = self._retry_checkpoint(current, goal, targets, loads, review_coverage)
        result = await self._run_retry_checkpoint(db, goal, run, current, checkpoint)
        if result is not None:
            return result
        return await self._finalize_retry_checkpoint(db, goal, run, current, checkpoint)

    async def _finalize_retry_checkpoint(
        self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun,
        current: OrchestrationProcessRun, checkpoint: dict[str, Any],
    ) -> dict:
        reviews, findings, eligibility, semantic_assessments = [], [], {}, checkpoint["completed"]
        targets = []
        for target in checkpoint["targets"]:
            agent = await db.get(Agent, uuid.UUID(target["agent_id"]))
            if agent is None:
                raise ValueError(f"agent {target['agent_id']} no longer exists")
            assessment = copy.deepcopy(target["deterministic_assessment"])
            semantic = semantic_assessments[target["agent_id"]]
            assessment["approved_for_work_functions"] = [
                name for name in assessment["approved_for_work_functions"]
                if name in set(semantic["approved_work_functions"])
            ]
            load = target["load_snapshot"]
            review = await self.review_service.create_review(
                db, goal.id, agent_id=agent.id, fit_summary=assessment["fit_summary"],
                review_context=json.dumps({"active_session_count": load["active_sessions"],
                    "active_task_count": load["active_tasks"], "available_outcome_hint_count": load["outcome_hints"],
                    "goal": target["goal_snapshot"], "semantic_assessment": semantic,
                    "deterministic_findings": assessment["warnings"], "orchestration_model": checkpoint.get("model"),
                    "process_version": PROCESS_VERSION}, sort_keys=True, separators=(",", ":")),
                proposed_work_functions=assessment["proposed_work_functions"], strengths=assessment["strengths"],
                risks=assessment["risks"], recommended_changes=assessment["recommended_changes"],
                approved_for_work_functions=assessment["approved_for_work_functions"], run_id=run.id,
                source_process_run_id=current.id, definition_snapshot=target["agent_snapshot"],
            )
            reviews.append(review)
            targets.append((agent, target["candidate_work_functions"], False))
            eligibility[target["agent_id"]] = list(review.approved_for_work_functions)
            findings.extend((agent, review, warning) for warning in assessment["warnings"])
        await self._resolve_superseded_warnings(db, goal.id, current.id)
        await self._clear_recovered_failure(db, goal, run, checkpoint)
        warnings = [await self.warning_service.create_warning(
            db, goal.id, warning_type=finding["warning_type"], severity=finding["severity"], message=finding["message"],
            run_id=run.id, source_process_run_id=current.id, related_agent_id=agent.id, source_agent_review_id=review.id,
        ) for agent, review, finding in findings]
        decision_ids = []
        for agent, _proposed, _inactive in targets:
            semantic = semantic_assessments[str(agent.id)]
            if semantic["status"] != "improvement_proposed":
                continue
            review = next(item for item in reviews if item.agent_id == agent.id)
            target = next(item for item in checkpoint["targets"] if item["agent_id"] == str(agent.id))
            decision = await self.decision_service.create_pending(
                db, goal.id, decision_key=f"{DECISION_KEY_PREFIX}{agent.id}",
                title=f"Review proposed definition for {agent.name}",
                question="Approve, reject, or edit the proposed agent description and persona?",
                authority="human", options=["approve", "reject", "edit"],
                context=json.dumps({"agent_id": str(agent.id), "affected_work_functions": review.proposed_work_functions,
                    "problems": semantic["problems"], "reason": semantic["reason"],
                    "original_description": target["agent_snapshot"]["description"],
                    "proposed_description": semantic["proposed_description"],
                    "original_persona": target["agent_snapshot"]["system_prompt"],
                    "proposed_persona": semantic["proposed_persona"],
                    "deterministic_warnings": target["deterministic_assessment"]["warnings"]}, sort_keys=True),
                recommendation="approve",
                consequences=(
                    "Approve applies only the proposed text; reject preserves the definition; "
                    "edit applies the supplied description and persona. The batch completes this review."
                ),
                run_id=run.id, source_process_run_id=current.id,
            )
            decision_ids.append(str(decision.id))
        coverage = checkpoint["coverage_fingerprint"]
        outputs = self._outputs(checkpoint["fingerprint"], coverage, reviews, targets,
            eligibility, warnings, semantic_assessments,
            sorted(set((current.outputs or {}).get("decision_ids", []) + decision_ids)))
        from huddleroom.services.orchestration_service import OrchestrationService

        if not run.active_blockers:
            if goal.status == "blocked":
                goal.status = "active"
            if run.status == "blocked":
                run.status = "running"
        if decision_ids:
            current.outputs = outputs
            await db.flush()
            await self.process_service.park_process(db, current)
            return self._summary("waiting_decision")
        result = await self._complete_review(db, goal, run, current, reviews, warnings, outputs)
        await db.flush()
        return result

    @staticmethod
    def _outputs(
        fingerprint: str,
        coverage_fingerprint: str,
        reviews: list[OrchestrationAgentReview],
        targets: list[tuple[Agent, list[str], bool]],
        eligibility: dict[str, list[str]],
        warnings,
        semantic_assessments: dict[str, dict[str, Any]],
        decision_ids: list[str],
    ) -> dict[str, Any]:
        return {
            "fingerprint": fingerprint,
            "coverage_fingerprint": coverage_fingerprint,
            "review_ids": [str(review.id) for review in reviews],
            "target_ids": [str(agent.id) for agent, _proposed, _inactive in targets],
            "eligibility": eligibility,
            "warning_ids": [str(warning.id) for warning in warnings],
            "warning_count": len(warnings),
            "compressed": False,
            "semantic_assessments": semantic_assessments,
            "decision_ids": decision_ids,
            "orchestration_model": settings.orchestration_model,
            "process_version": PROCESS_VERSION,
            "deterministic_policy": DETERMINISTIC_POLICY,
            "gates": {"agent_definitions_reviewed": True},
        }

    async def _complete_review(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
        current: OrchestrationProcessRun,
        reviews: list[OrchestrationAgentReview],
        warnings,
        outputs: dict[str, Any],
    ) -> dict:
        compressed = goal.weight == "trivial"
        if compressed:
            outputs["compressed"] = True
        await self.memory_service.upsert_section(
            db,
            goal.project_id,
            goal.id,
            section_key=MEMORY_SECTION_KEY,
            title="Agent definition review",
            body=self._memory_body(reviews, warnings, compressed=compressed),
            summary=f"Reviewed {len(reviews)} agent definitions; {len(warnings)} warnings.",
            toc_order=MEMORY_TOC_ORDER,
            run_id=run.id,
            created_by="orchestrator:agent_definition_review",
        )
        await self.process_service.complete_process(db, current, outputs=outputs)
        return self._summary("completed")

    async def _complete_persisted_review(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
        current: OrchestrationProcessRun,
    ) -> dict:
        reviews = [
            review
            for review in await self.review_service.list_reviews(db, goal.id)
            if review.source_process_run_id == current.id
        ]
        warnings = [
            warning
            for warning in await self.warning_service.list_warnings(db, goal.id)
            if warning.source_process_run_id == current.id
        ]
        outputs = dict(current.outputs or {})
        outputs["decision_ids"] = sorted(set(outputs.get("decision_ids", [])))
        return await self._complete_review(db, goal, run, current, reviews, warnings, outputs)

    async def handle_skip(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        skipped_run: OrchestrationProcessRun,
    ) -> None:
        await self.warning_service.create_warning(
            db,
            goal.id,
            warning_type=SKIP_WARNING_TYPE,
            severity="warning",
            message=SKIP_WARNING_MESSAGE,
            run_id=skipped_run.run_id,
            source_process_run_id=skipped_run.id,
        )
        await self.memory_service.upsert_section(
            db,
            goal.project_id,
            goal.id,
            section_key=MEMORY_SECTION_KEY,
            title="Agent definition review",
            body=SKIP_WARNING_MESSAGE,
            summary="Agent definition review was skipped by the human.",
            toc_order=MEMORY_TOC_ORDER,
            run_id=skipped_run.run_id,
            created_by="orchestrator:agent_definition_review",
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
        """Write the accepted snapshot of an approved recovery configuration.

        A superseding, synchronously completed ``approved_recovery`` row: no semantic
        review, no waiting_decision, no definition approval card. The caller owns the
        transaction (the roster-first recovery aggregate). Everything is validated
        before the first write.
        """
        payload = RecoveryDefinitionAcceptanceV1.model_validate(accepted_payload)
        evidence = tuple(EvidenceV1.model_validate(item) for item in evidence)
        if goal.weight == "trivial" or run.goal_id != goal.id:
            raise ValueError("unsupported_recovery_goal")
        provenance = await self.process_service.recovery_provenance(
            db, goal.id, proposal_id=proposal_id, proposal_hash=proposal_hash,
            authority_decision_id=authority_decision_id, call_id=payload.call_id,
            canonical_request_hash=payload.canonical_request_hash,
        )
        agents, assessments = {}, {}
        for agent_id, functions in payload.eligibility.items():
            agent = await db.get(Agent, agent_id)
            if agent is None:
                raise ValueError("inactive_or_unknown_agent")
            agents[agent_id] = agent
            assessments[agent_id] = recovery_eligibility_assessment(agent, functions)
        previous = await self.process_service.get_current(db, goal.id, PROCESS_TYPE)
        previous_outputs = (previous.outputs or {}) if previous is not None and previous.status == "completed" else {}
        previous_review_ids = {str(item) for item in previous_outputs.get("review_ids", [])}
        accepted_reviews = {
            review.agent_id: review for review in await self.review_service.list_reviews(db, goal.id)
            if str(review.id) in previous_review_ids and review.agent_id is not None
        }

        # -- writes start here --------------------------------------------------------------
        fingerprint = await self.current_fingerprint(db, goal, run)
        current = await self.process_service.start_process(
            db, goal.id, process_type=PROCESS_TYPE, trigger_reason=RECOVERY_TRIGGER_REASON,
            run_id=run.id, input_snapshot={"fingerprint": fingerprint},
            process_version=PROCESS_VERSION, supersede_waiting=True, supersede_running=True,
        )
        if current.trigger_reason != RECOVERY_TRIGGER_REASON or current.status != "running":
            raise ValueError("recovery_snapshot_not_started")
        for decision in await self._proposal_decisions(db, goal.id):
            if decision.status == "pending":
                await self.decision_service.cancel_decision(
                    db, decision, reason="superseded by an approved recovery configuration",
                )
        changed = set(payload.changed_agent_ids)
        evidence_payload = [item.model_dump(mode="json") for item in evidence]
        kept, reviews, findings = [], [], []
        for agent_id, functions in payload.eligibility.items():
            agent, prior = agents[agent_id], accepted_reviews.get(agent_id)
            if (
                agent_id not in changed and prior is not None
                and sorted(prior.approved_for_work_functions or []) == sorted(functions)
                and self._snapshot_matches(prior.definition_snapshot, agent)
            ):
                kept.append(prior)  # unchanged definition: carry the accepted eligibility forward
                continue
            assessment = assessments[agent_id]
            review = await self.review_service.create_review(
                db, goal.id, agent_id=agent.id,
                fit_summary=(
                    f"Approved recovery configuration: accepted for {', '.join(functions)}. "
                    "No standard definition review was run."
                ),
                review_context=json.dumps({
                    "source": RECOVERY_TRIGGER_REASON, **provenance,
                    "recovery_evidence": evidence_payload,
                    "deterministic_findings": assessment["warnings"],
                    "process_version": PROCESS_VERSION,
                }, sort_keys=True, separators=(",", ":")),
                proposed_work_functions=list(functions), strengths=assessment["strengths"],
                risks=assessment["risks"], recommended_changes=assessment["recommended_changes"],
                approved_for_work_functions=list(functions), run_id=run.id,
                source_process_run_id=current.id,
            )
            reviews.append(review)
            findings.extend((agent, review, warning) for warning in assessment["warnings"])
        await self._resolve_superseded_warnings(
            db, goal.id, current.id, keep_review_ids={review.id for review in kept},
            reason="superseded by an approved recovery configuration",
        )
        warnings = [await self.warning_service.create_warning(
            db, goal.id, warning_type=finding["warning_type"], severity=finding["severity"],
            message=finding["message"], run_id=run.id, source_process_run_id=current.id,
            related_agent_id=agent.id, source_agent_review_id=review.id,
        ) for agent, review, finding in findings]
        eligibility = {str(agent_id): list(functions) for agent_id, functions in payload.eligibility.items()}
        target_ids = sorted(set(eligibility) | {str(item) for item in previous_outputs.get("target_ids", [])})
        accepted = kept + reviews
        outputs = {
            "fingerprint": fingerprint,
            "coverage_fingerprint": await self.current_coverage_fingerprint(
                db, goal, run, covered_target_ids=set(target_ids),
            ),
            "review_ids": [str(review.id) for review in accepted],
            "target_ids": target_ids,
            "eligibility": eligibility,
            "warning_ids": [str(warning.id) for warning in warnings],
            "warning_count": len(warnings),
            "compressed": False,
            "semantic_assessments": {},
            "decision_ids": [],
            "process_version": PROCESS_VERSION,
            "deterministic_policy": DETERMINISTIC_POLICY,
            "gates": {"agent_definitions_reviewed": True},
            "recovery_provenance": {**provenance, "evidence_labels": [item.label for item in evidence]},
        }
        lines = [
            f"Accepted through approved recovery (proposal {proposal_id}); no standard definition review was run.",
            f"Accepted agent definitions: {len(accepted)} ({len(reviews)} re-assessed, {len(kept)} carried forward)",
            f"Active findings: {len(warnings)}",
            *(f"- agent:{agent_id}: eligible for {', '.join(functions)}" for agent_id, functions in eligibility.items()),
        ]
        await self.memory_service.upsert_section(
            db, goal.project_id, goal.id, section_key=MEMORY_SECTION_KEY, title="Agent definition review",
            body="\n".join(lines),
            summary=f"Approved recovery accepted {len(accepted)} agent definitions; {len(warnings)} warnings.",
            toc_order=MEMORY_TOC_ORDER, run_id=run.id, created_by="orchestrator:agent_definition_review",
        )
        return await self.process_service.complete_process(db, current, outputs=outputs)

    def _applied_text(self, current: OrchestrationProcessRun, decisions) -> dict[str, tuple[str, str]]:
        """Description/persona this process's answered approve/edit decisions applied, by agent id."""
        applied = {}
        for decision in decisions:
            if (
                decision.source_process_run_id != current.id or decision.status != "answered"
                or decision.selected_option not in {"approve", "edit"}
            ):
                continue
            context = self._decision_context(decision)
            description, persona = context.get("proposed_description"), context.get("proposed_persona")
            if decision.selected_option == "edit":
                try:
                    edited = json.loads(decision.reason or "{}")
                except (json.JSONDecodeError, TypeError):
                    continue
                description, persona = edited.get("description"), edited.get("persona")
            if isinstance(description, str) and isinstance(persona, str):
                applied[context.get("agent_id")] = (description.strip(), persona.strip())
        return applied

    async def _refresh_deterministic_eligibility(
        self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun,
        current: OrchestrationProcessRun, decisions,
    ) -> bool:
        """Re-run only the deterministic rules for an old-policy completed review.

        Trusts the current text only when it matches the review snapshot or the text this
        process's own answered decision applied; any other drift is left to the normal
        semantic-rerun path (no eligibility is granted here). No analyzer call.
        """
        outputs = dict(current.outputs or {})
        review_ids = {str(item) for item in outputs.get("review_ids", [])}
        reviews = [
            review for review in await self.review_service.list_reviews(db, goal.id)
            if str(review.id) in review_ids and review.source_process_run_id == current.id
        ]
        semantic = outputs.get("semantic_assessments") or {}
        applied = self._applied_text(current, decisions)
        plan = []
        for review in reviews:
            agent = await db.get(Agent, review.agent_id, populate_existing=True) if review.agent_id else None
            if agent is None or str(agent.id) not in semantic:
                return False
            snapshot, live = review.definition_snapshot or {}, self.review_service._build_snapshot(agent)  # pylint: disable=protected-access
            text_applied = applied.get(str(agent.id)) == ((agent.description or "").strip(), (agent.system_prompt or "").strip())
            skipped = {"description", "system_prompt"} if text_applied else set()
            if any(snapshot.get(f) != live.get(f) for f in FINGERPRINT_SNAPSHOT_FIELDS if f not in skipped):
                return False
            plan.append((review, agent, assess_agent_definition(agent, list(review.proposed_work_functions or []))))
        if not plan:
            return False
        # -- writes start here (no awaits that can fail halfway besides warning creation) --
        eligibility = dict(outputs.get("eligibility") or {})
        for review, agent, assessment in plan:
            allowed = set(semantic[str(agent.id)].get("approved_work_functions", []))
            review.approved_for_work_functions = [
                name for name in assessment["approved_for_work_functions"] if name in allowed
            ]
            review.fit_summary = assessment["fit_summary"]
            review.strengths, review.risks = assessment["strengths"], assessment["risks"]
            review.recommended_changes = assessment["recommended_changes"]
            eligibility[str(agent.id)] = list(review.approved_for_work_functions)
        plan_ids = {review.id for review, _agent, _assessment in plan}
        for warning in await self.warning_service.list_warnings(db, goal.id, active_only=True):
            if warning.source_agent_review_id in plan_ids:
                await self.warning_service.resolve_warning(
                    db, warning, resolved_by="orchestrator:agent_definition_review",
                    reason="deterministic review policy refreshed",
                )
        warnings = [
            await self.warning_service.create_warning(
                db, goal.id, warning_type=item["warning_type"], severity=item["severity"], message=item["message"],
                run_id=run.id, source_process_run_id=current.id, related_agent_id=agent.id,
                source_agent_review_id=review.id,
            )
            for review, agent, assessment in plan for item in assessment["warnings"]
        ]
        outputs.update(
            eligibility=eligibility, deterministic_policy=DETERMINISTIC_POLICY,
            warning_ids=[str(item.id) for item in warnings], warning_count=len(warnings),
        )
        current.outputs = outputs
        await db.flush()
        return True

    def _snapshot_matches(self, snapshot: dict | None, agent: Agent) -> bool:
        """The accepted snapshot still describes the live definition (fingerprint fields)."""
        live = self.review_service._build_snapshot(agent)  # pylint: disable=protected-access
        snapshot = snapshot if isinstance(snapshot, dict) else {}
        return all(snapshot.get(field) == live.get(field) for field in FINGERPRINT_SNAPSHOT_FIELDS)

    async def _current_run_assignments(
        self, db: AsyncSession, project_id, run_id
    ) -> list[tuple[Task, Agent, str]]:
        authority_decision_ids = {
            str(action_id): str(decision_id)
            for decision_id, action_id in (
                await db.execute(
                    select(
                        OrchestrationAuthorityDecision.id,
                        OrchestrationAuthorityDecision.related_action_id,
                    ).where(
                        OrchestrationAuthorityDecision.run_id == run_id,
                        OrchestrationAuthorityDecision.related_action_id.is_not(None),
                    )
                )
            ).all()
        }
        authority_delivery_action_ids = {
            str(action_id)
            for action_id in (
                await db.scalars(
                    select(OrchestrationAction.id).where(
                        OrchestrationAction.run_id == run_id,
                        OrchestrationAction.idempotency_key.startswith("authority_decision_delegate:"),
                    )
                )
            ).all()
        }
        tasks = list(
            (
                await db.execute(
                    select(Task).where(
                        Task.project_id == project_id,
                        Task.assigned_to.is_not(None),
                    )
                )
            ).scalars()
        )
        assignments = []
        for task in tasks:
            orchestration = (task.metadata_ or {}).get("orchestration", {})
            if not isinstance(orchestration, dict) or orchestration.get("run_id") != str(run_id):
                continue
            agent = await db.get(Agent, task.assigned_to)
            if agent is None:
                continue
            work_function = str(orchestration.get("work_function") or "").strip()
            authority_decision_id = (
                orchestration.get("authority_decision_id")
                or authority_decision_ids.get(orchestration.get("action_id"))
            )
            if authority_decision_id or orchestration.get("action_id") in authority_delivery_action_ids:
                continue
            assignments.append((task, agent, work_function))
        return sorted(assignments, key=lambda item: str(item[0].id))

    async def _review_inputs(
        self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun
    ) -> tuple[
        list[tuple[Task, Agent, str]],
        list[tuple[Agent, list[str], bool]],
        str,
    ]:
        assignments = await self._current_run_assignments(db, goal.project_id, run.id)
        targets = await self._targets(db, goal, assignments)
        return assignments, targets, self._fingerprint(goal, assignments, targets)

    async def current_fingerprint(
        self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun
    ) -> str:
        """Return the side-effect-free fingerprint for the run's current inputs."""
        _assignments, _targets, fingerprint = await self._review_inputs(db, goal, run)
        return fingerprint

    async def current_coverage_fingerprint(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
        *,
        covered_target_ids: set[str] | None = None,
    ) -> str:
        """Return material review coverage, excluding assignment identity and count."""
        assignments, targets, _fingerprint = await self._review_inputs(db, goal, run)
        if covered_target_ids is None:
            return self._coverage_fingerprint(goal, targets)
        return await self._coverage_fingerprint_for(db, goal, assignments, covered_target_ids)

    async def _coverage_fingerprint_for(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        assignments: list[tuple[Task, Agent, str]],
        covered_target_ids: set[str],
        *,
        fields: tuple[str, ...] = FINGERPRINT_SNAPSHOT_FIELDS,
    ) -> str:
        """Canonical coverage formula over already-loaded assignments."""
        baseline_assignments = [
            assignment
            for assignment in assignments
            if not self._assignment_action_id(assignment[0])
        ]
        coverage_targets = await self._targets(db, goal, baseline_assignments)
        # Plan/delegation-action assignments must never contribute a work
        # function to coverage (only baseline assignments may). But an agent
        # that was reviewed and covered, and now only appears via an
        # action-tagged assignment (its baseline assignment was replaced),
        # must stay represented in coverage_targets -- otherwise it drops out
        # of the payload and trips a false readiness 409. Strip the
        # work_function on action-tagged assignments instead of excluding
        # them, so the agent's presence/definition_snapshot survive the merge
        # without importing the action's work_function.
        sanitized_assignments = [
            (task, agent, "" if self._assignment_action_id(task) else work_function)
            for task, agent, work_function in assignments
        ]
        sanitized_targets = await self._targets(db, goal, sanitized_assignments)
        coverage_targets = self._merge_coverage_targets(
            coverage_targets,
            [
                target
                for target in sanitized_targets
                if str(target[0].id) in covered_target_ids
            ],
        )
        return self._coverage_fingerprint(goal, coverage_targets, fields=fields)

    @staticmethod
    def _merge_coverage_targets(
        *target_groups: list[tuple[Agent, list[str], bool]],
    ) -> list[tuple[Agent, list[str], bool]]:
        agents: dict[Any, Agent] = {}
        proposed_by_agent: dict[Any, set[str]] = {}
        inactive_by_agent: dict[Any, bool] = {}
        for targets in target_groups:
            for agent, proposed, inactive in targets:
                agents[agent.id] = agent
                proposed_by_agent.setdefault(agent.id, set()).update(proposed)
                inactive_by_agent[agent.id] = inactive_by_agent.get(agent.id, False) or inactive
        return [
            (
                agent,
                sorted(proposed_by_agent.get(agent.id, set())),
                inactive_by_agent.get(agent.id, False),
            )
            for agent in sorted(agents.values(), key=lambda value: str(value.id))
        ]

    @staticmethod
    def _assignment_action_id(task: Task) -> str | None:
        orchestration = (task.metadata_ or {}).get("orchestration", {})
        if not isinstance(orchestration, dict):
            return None
        action_id = orchestration.get("action_id")
        return str(action_id) if action_id else None

    async def _targets(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        assignments: list[tuple[Task, Agent, str]],
    ) -> list[tuple[Agent, list[str], bool]]:
        manager = (
            await db.get(Agent, goal.manager_agent_id)
            if goal.weight == "standard" and goal.authority_model == "agent_manager"
            and goal.manager_agent_id is not None
            else None
        )
        active = (
            list((await db.execute(select(Agent).where(Agent.is_active.is_(True)))).scalars())
            if goal.weight in ("standard", "substantial")
            else []
        )
        # ponytail: local import avoids a module-level cycle with team_hierarchy
        from huddleroom.services.orchestration_team_hierarchy import infer_required_work_functions

        return coverage_targets_from_projection(
            goal, assignments, active_agents=active, manager=manager,
            roster_mapper=self.roster_mapper,
            required_functions=(
                infer_required_work_functions(goal) if goal.weight == "standard" else ()
            ),
        )

    @staticmethod
    def _fingerprint(
        goal: OrchestrationGoal,
        assignments: list[tuple[Task, Agent, str]],
        targets: list[tuple[Agent, list[str], bool]],
        *,
        fields: tuple[str, ...] = FINGERPRINT_SNAPSHOT_FIELDS,
    ) -> str:
        payload = {
            "weight": goal.weight,
            "objective": goal.objective,
            "success_criteria": copy.deepcopy(goal.success_criteria or []),
            "constraints": copy.deepcopy(goal.constraints or {}),
            "assignments": sorted(
                (str(task.id), str(agent.id), work_function)
                for task, agent, work_function in assignments
            ),
            "authority_model": goal.authority_model,
            "selected_manager": (
                str(goal.manager_agent_id or goal.manager_user_id)
                if goal.manager_agent_id or goal.manager_user_id
                else None
            ),
            "targets": [
                {
                    "agent_id": str(agent.id),
                    "proposed_work_functions": proposed,
                    "definition_snapshot": {
                        field: copy.deepcopy(getattr(agent, field))
                        for field in fields
                    },
                }
                for agent, proposed, _inactive in targets
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode()).hexdigest()

    @staticmethod
    def _coverage_fingerprint(
        goal: OrchestrationGoal,
        targets: list[tuple[Agent, list[str], bool]],
        *,
        fields: tuple[str, ...] = FINGERPRINT_SNAPSHOT_FIELDS,
    ) -> str:
        return coverage_fingerprint_from_projection(goal, targets, fields=fields)

    async def _resolve_superseded_warnings(
        self, db: AsyncSession, goal_id, current_process_id, *, keep_review_ids=frozenset(),
        reason: str = "superseded by a successful agent definition review",
    ) -> None:
        process_ids = set(
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
        if not process_ids:
            return
        review_ids = set(
            (
                await db.execute(
                    select(OrchestrationAgentReview.id).where(
                        OrchestrationAgentReview.source_process_run_id.in_(process_ids)
                    )
                )
            ).scalars()
        ) - set(keep_review_ids)
        for warning in await self.warning_service.list_warnings(db, goal_id, active_only=True):
            linked_review = warning.source_agent_review_id in review_ids
            prior_skip = (
                warning.source_process_run_id in process_ids
                and warning.warning_type in {f"{PROCESS_TYPE}_skipped", SKIP_WARNING_TYPE}
            )
            if linked_review or prior_skip:
                await self.warning_service.resolve_warning(
                    db,
                    warning,
                    resolved_by="orchestrator:agent_definition_review",
                    reason=reason,
                )

    @staticmethod
    def _memory_body(reviews, warnings, *, compressed: bool) -> str:
        if compressed:
            return "Trivial goal: agent definition review compressed; no agents reviewed."
        lines = [f"Reviewed agent definitions: {len(reviews)}", f"Active findings: {len(warnings)}"]
        for review in reviews:
            lines.append(
                f"- agent:{review.agent_id}: eligible for "
                f"{', '.join(review.approved_for_work_functions) or 'no proposed functions'}"
            )
        return "\n".join(lines)

    @staticmethod
    def _summary(status: str) -> dict:
        return {
            "process_type": PROCESS_TYPE,
            "status": status,
            "questions_created": 0,
        }


async def accepted_work_functions(db: AsyncSession, goal_id) -> dict[str, list[str]] | None:
    """Per-agent approved work functions from the current review; None when not reviewed."""
    current = await OrchestrationProcessService().get_current(db, goal_id, PROCESS_TYPE)
    if current is None or current.status == "skipped":
        return None
    eligibility = (current.outputs or {}).get("eligibility")
    return eligibility if isinstance(eligibility, dict) else None


async def pending_definition_repair(db: AsyncSession, goal_id, agent_id) -> OrchestrationAuthorityDecision | None:
    """The pending approve/reject/edit definition decision for an agent, if any."""
    return await OrchestrationAuthorityDecisionService().get_pending_by_key(
        db, goal_id, f"{DECISION_KEY_PREFIX}{agent_id}"
    )
