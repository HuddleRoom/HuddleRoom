"""Work-function inference and LLM-orchestrated team hierarchy drafting."""
from __future__ import annotations

import hashlib
import json
import re
import uuid

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
from huddleroom.services.agent_service import AgentService
from huddleroom.services.orchestration_agent_review_service import OrchestrationAgentReviewService
from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
from huddleroom.services.orchestration_goal_definition import requires_independent_verification
from huddleroom.services.orchestration_memory_service import OrchestrationMemoryService
from huddleroom.services.orchestration_process_service import OrchestrationProcessService
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
# Sentinel input_snapshot fingerprints used internally to force a real
# rerun in _advance_orchestrated's mismatch branch -- both mark a
# deliberate continuation of a decision the human is already mid-way
# through resolving (one proposal answered, or the fully-resolved hierarchy
# failed validation), not background input drift. Never the item-3
# one-time-suggestion path.
PROPOSAL_RESOLUTION_RERUN_SENTINEL = "rerun after proposal resolution"
RESOLVED_VALIDATION_FAILED_RERUN_SENTINEL = "rerun after resolved hierarchy validation failed"
HIERARCHY_CHANGES_REQUESTED_RERUN_SENTINEL = "rerun after hierarchy changes requested"
INTERNAL_RERUN_SENTINELS = frozenset(
    {
        PROPOSAL_RESOLUTION_RERUN_SENTINEL,
        RESOLVED_VALIDATION_FAILED_RERUN_SENTINEL,
        HIERARCHY_CHANGES_REQUESTED_RERUN_SENTINEL,
    }
)
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
_STANDARD_TIER_BLOCKING_GAPS = frozenset({
    "no_independent_verifier",
    "agent_definition_review_skipped",
})


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
    if not required.intersection(_SPECIALIST_FUNCTIONS):
        required.add("implementation")
    return [name for name in WORK_FUNCTION_ORDER if name in required]


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
            return await self._advance_orchestrated(db, goal, run, current)

        proposal = await self.build_proposal(db, goal, run)
        fingerprint = self._fingerprint(goal, proposal)
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
            current.input_snapshot.get("fingerprint")
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
                await self.warning_service.suggest_stale_inputs(
                    db,
                    goal.id,
                    process_type=PROCESS_TYPE,
                    process_run_id=current.id,
                    step_label="team hierarchy",
                    run_id=run.id,
                )
                return self._summary("completed")
            # Post-baseline: execution-time task assignments are part of the
            # fingerprint and must keep coverage in sync silently -- a human
            # is no longer actively reviewing the baseline, so restore the
            # original auto-rerun instead of leaving a stale suggestion.
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
            current.input_snapshot = {"fingerprint": fingerprint}
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
        warnings = await self._write_warnings(db, goal, run, current, proposal)
        await self._write_memory(db, goal, run, proposal)
        if goal.weight == "standard" and not (
            set(routing_gaps) & _STANDARD_TIER_BLOCKING_GAPS
        ):
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
    ) -> dict:
        payload = await self._analysis_input(db, goal)
        input_fingerprint = self._semantic_fingerprint(self._strip_volatile(payload))
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
                current.input_snapshot.get("fingerprint")
                if isinstance(current.input_snapshot, dict)
                else None
            )
            pure_version_bump = stored_fingerprint == old_formula_fingerprint
            current.process_version = PROCESS_VERSION
            if pure_version_bump:
                current.input_snapshot = {"fingerprint": input_fingerprint}
            # else: genuine drift co-occurring with the version bump -- the
            # version is bumped now so it isn't rechecked every tick, but the
            # stored fingerprint is left stale so the normal stale-inputs
            # path below fires (suggestion in baseline / auto-rerun
            # post-baseline).
            await db.flush()
        if (
            current is not None and current.status == "completed"
            and current.input_snapshot.get("fingerprint") == input_fingerprint
            and current.outputs.get("approval") in {"approve", "approve_with_documented_gaps"}
        ):
            return self._summary("completed")
        if current is not None and current.status == "completed":
            if run.phase == "baseline":
                # Baseline phase: stale inputs get a one-time suggestion, no
                # auto-rerun.
                await self.warning_service.suggest_stale_inputs(
                    db,
                    goal.id,
                    process_type=PROCESS_TYPE,
                    process_run_id=current.id,
                    step_label="team hierarchy",
                    run_id=run.id,
                )
                return self._summary("completed")
            # Post-baseline: execution-time task assignments are part of the
            # fingerprint and must keep coverage in sync silently -- a human
            # is no longer actively reviewing the baseline, so restore the
            # original auto-rerun instead of leaving a stale suggestion.
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
            and current.input_snapshot.get("fingerprint") != input_fingerprint
            and current.input_snapshot.get("fingerprint") not in INTERNAL_RERUN_SENTINELS
        ):
            # Stale inputs while a decision is pending: park stays parked,
            # one-time suggestion instead of cancel+restart (item 3). The
            # sentinel marker (deliberate batch-continuation rerun, not
            # background drift) is excluded and falls through below.
            await self.warning_service.suggest_stale_inputs(
                db,
                goal.id,
                process_type=PROCESS_TYPE,
                process_run_id=current.id,
                step_label="team hierarchy",
                run_id=run.id,
            )
            return self._summary("waiting_decision")
        elif current.input_snapshot.get("fingerprint") != input_fingerprint:
            await self._cancel_pending_for_run(db, goal.id, current.id)
            current = await self.process_service.start_process(
                db, goal.id, process_type=PROCESS_TYPE,
                trigger_reason="auto-rerun: hierarchy inputs changed",
                run_id=run.id, input_snapshot={"fingerprint": input_fingerprint},
                process_version=PROCESS_VERSION,
                supersede_waiting=True,
            )
        if current.status not in {"running", "waiting_decision"}:
            return self._summary(current.status)

        analysis = current.outputs.get("analysis") if isinstance(current.outputs, dict) else None
        if not isinstance(analysis, dict):
            from huddleroom.models.project import Project
            _proj = await db.get(Project, goal.project_id)
            project_dict = {"name": _proj.name, "description": _proj.description} if _proj else None
            try:
                result = await self.analyzer.review(
                    payload, project=project_dict, project_id=goal.project_id,
                )
            except Exception as exc:  # analyzer normalizes provider and semantic failures
                safe_error = _safe_completion_error(exc)
                request = getattr(exc, "request", None) or TeamHierarchyAnalyzer.build_request(payload, project=project_dict)
                current.outputs = {
                    "_lm_retry": {"kind": PROCESS_TYPE, "version": 1, "request": redact_semantic_payload(request)},
                    "retryable": True,
                    "error": safe_error,
                }
                await db.flush()
                return {**self._summary(current.status), "retryable": True, "error": safe_error}
            analysis = result.to_dict()
            current.outputs = {
                "analysis": analysis,
                "fingerprint": self._semantic_fingerprint({"input": self._strip_volatile(payload), "analysis": analysis}),
            }
            await db.flush()

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
                proposal = proposal_by_key[decision.decision_key]
                agent = await self.agent_service.create(db, AgentCreate(**proposal["definition"]))
                created_id = str(agent.id)
                metadata["created_agent_id"] = created_id
                await db.execute(update(OrchestrationAuthorityDecision).where(
                    OrchestrationAuthorityDecision.id == decision.id
                ).values(reason=json.dumps(metadata, sort_keys=True)))
                decision.reason = json.dumps(metadata, sort_keys=True)
            created_by_ref[f"proposal:{decision.decision_key.rsplit(':', 1)[-1]}"] = created_id

        if created_by_ref:
            current.input_snapshot = {
                "fingerprint": self._semantic_fingerprint(self._strip_volatile(await self._analysis_input(db, goal)))
            }

        if len(proposal_decisions) == len(proposal_by_key) and all(
            decision.status == "answered" and decision.selected_option == "approve"
            for decision in proposal_decisions
        ):
            frozen = self._resolve_analysis(analysis, created_by_ref)
            refreshed_payload = await self._analysis_input(db, goal)
            try:
                parse_team_hierarchy_analysis({
                    "proposed_agents": [],
                    "assignments": frozen["assignments"],
                    "reporting_lines": frozen["reporting_lines"],
                    "documented_gaps": frozen["documented_gaps"],
                    "rationale": frozen["rationale"],
                    "self_review": frozen["self_review"],
                }, refreshed_payload)
            except ValueError as exc:
                current.outputs = {**current.outputs, "retryable": False, "semantic_error": str(exc)}
                await self._cancel_pending_for_run(db, goal.id, current.id)
                current.input_snapshot = {"fingerprint": RESOLVED_VALIDATION_FAILED_RERUN_SENTINEL}
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
                outputs = {
                    **selected["proposal"],
                    "fingerprint": current.input_snapshot["fingerprint"],
                    "approval": approval.selected_option,
                    "approval_decision_id": str(approval.id),
                    "gates": {
                        "team_structure_reviewed": True,
                        "required_work_functions_mapped": not selected["proposal"]["documented_gaps"],
                        "agent_definitions_reviewed": True,
                        "missing_capabilities_handled": not selected["proposal"]["documented_gaps"],
                        "independent_verification_possible_or_overridden": True,
                    },
                }
                if current.status == "waiting_decision":
                    await self.process_service.resume_process(db, current)
                await self.process_service.complete_process(db, current, outputs=outputs)
                return self._summary("completed")
            if approval is not None and approval.status == "answered":
                await self._cancel_pending_for_run(db, goal.id, current.id)
                current.input_snapshot = {"fingerprint": HIERARCHY_CHANGES_REQUESTED_RERUN_SENTINEL}
                await db.flush()
                return await self._advance_orchestrated(db, goal, run, current)
            if approval is None:
                gaps = list(frozen["documented_gaps"])
                approve_key = "approve_with_documented_gaps" if gaps else "approve"
                await self.decision_service.create_pending(
                    db, goal.id, decision_key=APPROVAL_DECISION_KEY,
                    title="Approve team hierarchy",
                    question="Approve the proposed team hierarchy and role assignments?",
                    authority="human", authority_agent_id=None,
                    options=[{"key": approve_key, "proposal": frozen}, {"key": "request_changes"}],
                    context=json.dumps({"documented_gaps": frozen["documented_gaps"]}, sort_keys=True),
                    recommendation=approve_key, run_id=run.id, source_process_run_id=current.id,
                )
                questions_created = 1
            else:
                questions_created = 0
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
            "approved_but_unused_agent_ids": sorted(approved - used),
        }

    async def _analysis_input(self, db: AsyncSession, goal: OrchestrationGoal) -> dict:
        agents = list((await db.execute(
            select(Agent).where(Agent.is_active.is_(True)).order_by(Agent.name, Agent.id)
        )).scalars())
        loads = await self.roster_mapper.loads_by_agent(db, goal.project_id)
        manager = self._manager(goal)
        if manager["kind"] == "agent":
            selected = next((agent for agent in agents if str(agent.id) == manager["id"]), None)
            manager = {
                **manager,
                "name": selected.name if selected else None,
                "role": selected.role if selected else None,
            }
        exclusions = []
        for decision in await self.decision_service.list_decisions(db, goal.id):
            if not (
                decision.status == "answered"
                and decision.decision_key.startswith("team_hierarchy:agent:")
                and decision.selected_option in {"edit", "reject"}
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
        return redact_semantic_payload({
            "schema_version": 1,
            "goal": {
                "id": str(goal.id), "objective": goal.objective,
                "success_criteria": goal.success_criteria, "constraints": goal.constraints,
                "weight": goal.weight,
            },
            "selected_manager": manager,
            "required_work_functions": infer_required_work_functions(goal),
            "agents": [
                {
                    "id": str(agent.id), "name": agent.name, "role": agent.role,
                    "description": agent.description, "system_prompt": agent.system_prompt,
                    "provider": agent.provider, "model": agent.model,
                    "adapter_type": agent.adapter_type, "cli_runtime": agent.cli_runtime,
                    "capabilities": list(agent.capabilities or []), "config": agent.config or {},
                    "workload": loads.get(agent.id).to_dict() if loads.get(agent.id) else {
                        "active_tasks": 0, "active_sessions": 0, "outcome_hint_count": 0, "penalty": 0,
                    },
                }
                for agent in agents
            ],
            "resolved_proposal_exclusions": exclusions,
        })

    @staticmethod
    def _semantic_fingerprint(payload: dict) -> str:
        return hashlib.sha256(json.dumps(
            redact_semantic_payload(payload), sort_keys=True, separators=(",", ":"), default=str
        ).encode()).hexdigest()

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
            stripped_payload = self._strip_volatile(payload)
            return self._semantic_fingerprint(
                {"input": stripped_payload, "analysis": analysis}
                if isinstance(analysis, dict) else stripped_payload
            )
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
            result = await self.analyzer.review_request(request, project_id=goal.project_id)
        except Exception as exc:  # mirror advance(): re-park instead of surfacing a 500
            safe_error = _safe_completion_error(exc)
            current.outputs = {
                "_lm_retry": {"kind": PROCESS_TYPE, "version": 1, "request": redact_semantic_payload(request)},
                "retryable": True,
                "error": safe_error,
            }
            await db.flush()
            return {**self._summary(current.status), "retryable": True, "error": safe_error}
        payload = json.loads(request["messages"][1]["content"])
        analysis = result.to_dict()
        current.outputs = {
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
        explicit_independence = requires_independent_verification(
            goal.success_criteria, goal.objective
        )
        independent_functions = (
            ("review", "validation")
            if explicit_independence
            else ("validation",) if goal.weight == "substantial" else ()
        )
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
                if agent_id not in producer_ids and agent_id != goal.manager_agent_id
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
        payload = {
            "weight": goal.weight,
            "objective": goal.objective,
            "success_criteria": goal.success_criteria,
            "constraints": goal.constraints,
            "authority_model": goal.authority_model,
            "manager_agent_id": (
                str(goal.manager_agent_id) if goal.manager_agent_id else None
            ),
            "manager_user_id": (
                str(goal.manager_user_id) if goal.manager_user_id else None
            ),
            "source_agent_review_process_id": proposal[
                "source_agent_review_process_id"
            ],
            "review_ids": proposal["review_ids"],
            "required_work_functions": proposal["required_work_functions"],
            "role_to_agent": proposal["role_to_agent"],
            "candidate_agent_ids_by_function": {
                work_function: sorted(
                    set(proposal["candidate_agents"].get(work_function, []))
                )
                for work_function in proposal["required_work_functions"]
            },
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str
        )
        return hashlib.sha256(encoded.encode()).hexdigest()

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
                "gates": {
                    "team_structure_reviewed": True,
                    "required_work_functions_mapped": (
                        not outputs["missing_work_functions"] and not documented_gaps
                    ),
                    "agent_definitions_reviewed": agent_definitions_reviewed,
                    "missing_capabilities_handled": (
                        not outputs["missing_work_functions"] or documented_gaps
                    ),
                    "independent_verification_possible_or_overridden": (
                        outputs["independent_verification"]["possible"]
                        or documented_gaps
                    ),
                },
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
    ) -> None:
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
                and warning.warning_type
                in {"team_hierarchy_skipped", "team_hierarchy_not_reviewed"}
            ):
                await self.warning_service.resolve_warning(
                    db,
                    warning,
                    resolved_by="orchestrator:team_hierarchy",
                    reason="superseded by a successful team hierarchy review",
                )

    @staticmethod
    def _summary(status: str, questions_created: int = 0) -> dict:
        return {
            "process_type": PROCESS_TYPE,
            "status": status,
            "questions_created": questions_created,
        }
