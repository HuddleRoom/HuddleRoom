"""Pure deterministic assessment of an agent definition."""

# pylint: disable=too-many-locals

from __future__ import annotations

import copy
import hashlib
import json
import re
import uuid
from typing import Any

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
from huddleroom.services.orchestration_process_service import OrchestrationProcessService
from huddleroom.services.orchestration_roster_mapper import OrchestrationRosterMapper, PROFILES
from huddleroom.services.orchestration_warning_service import OrchestrationWarningService


CRITICAL_VALIDATION_FUNCTIONS = {"review", "validation"}
WEAK_REASONING_EFFORTS = {"none", "minimal", "low"}
UNSAFE_CLI_RUNTIMES = {"claude_code", "copilot", "opencode", "pi"}
PROCESS_TYPE = "agent_definition_review"
PROCESS_VERSION = 3
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

    effective_cli_runtime = config.get("cli_runtime", agent.cli_runtime or "claude_code")
    unsafe_permissions = config.get("allow_global_scope") is True or (
        agent.adapter_type == "cli" and effective_cli_runtime in UNSAFE_CLI_RUNTIMES
    )
    if unsafe_permissions:
        warnings.append(
            _warning(
                "agent_review_unsafe_permissions",
                "Agent explicitly has global scope or an unrestricted CLI runtime.",
            )
        )
        disqualified.update(normalized.values())
        recommended_changes.append("Restrict the agent's effective runtime permissions.")

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

    async def _apply_answered_proposals(
        self,
        db: AsyncSession,
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
        for decision in decisions:
            if decision.status != "answered" or str(decision.id) in processed:
                continue
            context = self._decision_context(decision)
            agent = await db.get(Agent, uuid.UUID(context["agent_id"]))
            if agent is None:
                raise ValueError(f"agent {context['agent_id']} no longer exists")
            if decision.selected_option in {"approve", "edit"}:
                description = context["proposed_description"]
                persona = context["proposed_persona"]
                if decision.selected_option == "edit":
                    edited = edits[decision.id]
                    description = edited.get("description")
                    persona = edited.get("persona")
                agent.description = description.strip()
                agent.system_prompt = persona.strip()
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

        _assignments, targets, fingerprint = await self._review_inputs(db, goal, run)
        coverage_fingerprint = self._coverage_fingerprint(goal, targets)
        proposal_decisions = await self._proposal_decisions(db, goal.id)
        stored_fingerprint = (
            current.input_snapshot.get("fingerprint")
            if current is not None and isinstance(current.input_snapshot, dict)
            else None
        )
        if (
            current is not None
            and current.status == "waiting_decision"
            and stored_fingerprint != fingerprint
        ):
            # Stale inputs while a decision is pending: never cancel the
            # pending decision or restart -- park stays parked, a one-time
            # suggestion is raised instead (item 3).
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
            proposal_changed = await self._apply_answered_proposals(db, current, answered)
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
            _assignments, refreshed_targets, refreshed_fingerprint = await self._review_inputs(
                db, goal, run
            )
            refreshed_coverage = self._coverage_fingerprint(goal, refreshed_targets)
            current.outputs = {
                **(current.outputs or {}),
                "fingerprint": refreshed_fingerprint,
                "coverage_fingerprint": refreshed_coverage,
            }
            await db.flush()
            return await self._complete_persisted_review(db, goal, run, current)
        legacy_missing_coverage_fingerprint = (
            current is not None
            and current.status == "completed"
            and current.process_version == PROCESS_VERSION
            and current.outputs.get("fingerprint") == fingerprint
            and current.outputs.get("coverage_fingerprint") is None
        )
        if legacy_missing_coverage_fingerprint:
            # Pre-coverage-fingerprint legacy row backfilling a field that
            # didn't exist yet: silently re-stamp, no warning/rerun. Falls
            # through to the normal fast path below, which will now match.
            current.outputs = {
                **(current.outputs or {}),
                "coverage_fingerprint": coverage_fingerprint,
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
            old_formula_fingerprint = self._fingerprint(goal, _assignments, targets, fields=SNAPSHOT_FIELDS)
            old_formula_coverage = self._coverage_fingerprint(goal, targets, fields=SNAPSHOT_FIELDS)
            stored_fp = current.outputs.get("fingerprint")
            stored_cov = current.outputs.get("coverage_fingerprint")
            pure_version_bump = stored_fp == old_formula_fingerprint and (
                stored_cov is None or stored_cov == old_formula_coverage
            )
            current.process_version = PROCESS_VERSION
            current.outputs = {
                **(current.outputs or {}),
                "process_version": PROCESS_VERSION,
            }
            if pure_version_bump:
                current.outputs = {
                    **(current.outputs or {}),
                    "fingerprint": fingerprint,
                    "coverage_fingerprint": coverage_fingerprint,
                }
            # else: genuine drift co-occurring with the version bump -- the
            # version is bumped now so it isn't rechecked every tick, but the
            # stored fingerprint/coverage_fingerprint are left stale so the
            # normal stale-inputs path below fires (suggestion in baseline
            # phase / auto-rerun post-baseline).
            await db.flush()
        if (
            current is not None
            and current.status == "completed"
            and current.process_version == PROCESS_VERSION
            and current.outputs.get("fingerprint") == fingerprint
            and current.outputs.get("coverage_fingerprint") == coverage_fingerprint
        ):
            return self._summary("completed")

        stored_fingerprint = (
            current.input_snapshot.get("fingerprint")
            if current is not None and isinstance(current.input_snapshot, dict)
            else None
        )
        running_inputs_changed = (
            current is not None
            and current.status == "running"
            and bool(stored_fingerprint)
            and stored_fingerprint != fingerprint
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
                await self.warning_service.suggest_stale_inputs(
                    db,
                    goal.id,
                    process_type=PROCESS_TYPE,
                    process_run_id=current.id,
                    step_label="agent definition review",
                    run_id=run.id,
                )
                return self._summary("completed")
            # Post-baseline: execution-time task assignments are part of
            # _fingerprint and must keep coverage in sync silently -- a human
            # is no longer actively reviewing the baseline, so restore the
            # original auto-rerun instead of leaving a stale suggestion.
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
        checkpoint = self._retry_checkpoint(current, goal, targets, loads, coverage_fingerprint)
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
            coverage_targets = targets
        else:
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
        return self._coverage_fingerprint(goal, coverage_targets)

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
            if goal.authority_model == "agent_manager" and goal.manager_agent_id is not None:
                manager = await db.get(Agent, goal.manager_agent_id)
                if manager is not None:
                    agents[manager.id] = manager
                    proposed_by_agent.setdefault(manager.id, set()).add("management")
        else:
            active = list(
                (await db.execute(select(Agent).where(Agent.is_active.is_(True)))).scalars()
            )
            for agent in active:
                agents[agent.id] = agent
                inferred = {
                    work_function
                    for work_function in PROFILES
                    if not self.roster_mapper.score_agent_definition(agent, work_function).weak
                }
                proposed_by_agent.setdefault(agent.id, set()).update(inferred)

        return [
            (
                agent,
                sorted(proposed_by_agent.get(agent.id, set())),
                agent.id in assigned_ids and not bool(agent.is_active),
            )
            for agent in sorted(agents.values(), key=lambda value: str(value.id))
        ]

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
        payload = {
            "weight": goal.weight,
            "objective": goal.objective,
            "success_criteria": copy.deepcopy(goal.success_criteria or []),
            "constraints": copy.deepcopy(goal.constraints or {}),
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

    async def _resolve_superseded_warnings(
        self, db: AsyncSession, goal_id, current_process_id
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
        )
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
                    reason="superseded by a successful agent definition review",
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
