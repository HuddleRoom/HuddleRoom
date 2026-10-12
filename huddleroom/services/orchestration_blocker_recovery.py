"""Durable recovery lifecycle: proposal, approval, apply, verify, one retry, lineage.

Approval records the answer and reserves the first apply wrapper under the origin goal lock.
Effects run afterwards (``resume``/``execute_next_call``), each fenced by the SPR 175/176 reset
reservation and the original run identity. Nothing here waits on a provider, delegated task,
event or human while holding a lock: a wait is stored and the call returns.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from types import SimpleNamespace
from typing import Literal
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.orchestration import (
    ACTIVE_RUN_STATUSES, OrchestrationAction, OrchestrationDecision, OrchestrationGoal, OrchestrationRun,
    OrchestrationWait,
)
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision, OrchestrationWarning
from huddleroom.models.task import Task
from huddleroom.schemas.orchestration_recovery import (
    CapabilityGapAssessmentV1, EvidenceV1, ProposalDraftV1, RecoveryProposalResponse, RecoveryProposalV1,
    RecoveryRetryDispatchContext, RecoveryTriggerV1, ToolReceiptV1, TransientAssessmentV1, VerificationResultV1,
)
from huddleroom.services import orchestration_recovery_registry as registry_module
from huddleroom.services.orchestration_authority_service import (
    RECOVERY_DECISION_KEY_PREFIX, RECOVERY_GAP_DECISION_KEY_PREFIX, OrchestrationAuthorityDecisionService,
)
from huddleroom.services.orchestration_baseline_runtime import require_baseline_available
from huddleroom.services.orchestration_decision_dispatcher import OrchestrationDecisionDispatcher
from huddleroom.services.orchestration_goal_definition import goal_objective_with_clarifications
from huddleroom.services.orchestration_llm_decision_adapter import _redact_secrets
from huddleroom.services.orchestration_process_service import OrchestrationProcessService
from huddleroom.services.orchestration_recovery_analyzer import MAX_RECOVERY_EVIDENCE_ROUNDS, RecoveryAnalyzer
from huddleroom.services.orchestration_recovery_configuration import RecoveryConfigurationService
from huddleroom.services.orchestration_recovery_evidence import (
    RECOVERY_EVIDENCE_MAX_BYTES, RecoveryContextBuilder, RecoveryEvidenceService, _bounded,
)
from huddleroom.services.orchestration_recovery_registry import RecoveryToolRegistry, _json_hash
from huddleroom.services.orchestration_roster_mutation import RosterStateChanged, holds_goal_lock
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.orchestration_steering import OrchestrationSteeringService, SteeringVersionsChanged
from huddleroom.services.orchestration_team_hierarchy import select_plan_author
from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

__all__ = [
    "MAX_RECOVERY_ATTEMPTS_WITHOUT_PROGRESS", "MAX_RECOVERY_EVIDENCE_ROUNDS", "OrchestrationBlockerRecoveryService",
    "RecoveryAnswerReservation", "RecoveryDiagnosisFailed", "RecoveryLifecycleStatus", "RecoveryProposalStale",
]

AFTER_APPROVAL = "HuddleRoom applies these updates, verifies the expected state, then retries the blocked work once."
# Code-owned loop bound (spec): not a model field or operator setting.
MAX_RECOVERY_ATTEMPTS_WITHOUT_PROGRESS = 3
ANALYSIS_FAILED = "analysis_failed"
ADMISSION_REJECTED = "baseline_repair_admission_rejected"
_NOT_A_REFUSAL = (ANALYSIS_FAILED, ADMISSION_REJECTED)
_WRAPPER_TYPES = ("recovery_apply", "recovery_verify", "recovery_retry")
_GAP_KEY_PREFIX = RECOVERY_GAP_DECISION_KEY_PREFIX
_WARNING_TYPE = registry_module.RECOVERY_WARNING_TYPE
_WAIT_OWNER = "recovery"
_FAILED_TASK_STATUSES = frozenset({"failed", "cancelled"})
_DELEGATION_TOOL = "orchestration.delegate_remediation"
_MAX_TRIGGER_EVIDENCE = 32
# Loop decisions that perform no work cannot show that a blocked loop progressed.
_NON_PROGRESS_ACTIONS = frozenset({"noop", "record_warning", "suggest_agent", "ask_human", "pause_run", "recover_blocker"})


async def _steering_seal(db: AsyncSession, goal, run) -> dict:
    """Steering versions and active direction ids at proposal time; apply fences against them."""
    snapshot = await OrchestrationSteeringService().context_snapshot(db, goal, run)
    if not snapshot:
        return {}
    return {"versions": snapshot["versions"],
            "active_directions": [{"request_id": item["request_id"]} for item in snapshot["active_directions"]]}


class RecoveryLifecycleStatus(StrEnum):
    PENDING_APPROVAL = "pending_approval"
    APPLYING = "applying"
    VERIFYING = "verifying"
    RETRYING = "retrying"
    RESOLVED = "resolved"
    REJECTED = "rejected"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class RecoveryAnswerReservation:
    decision_id: UUID
    proposal_action_id: UUID
    first_wrapper_action_id: UUID | None  # None for reject: nothing is reserved


class RecoveryProposalStale(Exception):
    """Approval drift: cancel, `state_changed` and fresh diagnosis are flushed; the caller commits, then 409s."""

    def __init__(self, proposal_action_id: UUID, decision_id: UUID) -> None:
        super().__init__("recovery proposal state changed")
        self.proposal_action_id = proposal_action_id
        self.decision_id = decision_id


class RecoveryDiagnosisFailed(Exception):
    """Diagnosis produced no usable outcome; the rejected exchange is flushed and the trigger stays reserved."""

    def __init__(self, trigger_action_id: UUID, exchange_id: UUID, code: str) -> None:
        super().__init__(code)
        self.trigger_action_id = trigger_action_id
        self.exchange_id = exchange_id
        self.code = code


class _RunFenced(HTTPException):
    """Reset/run-identity fence: always propagates as a 409, never handled as a step failure."""


def _trigger_key(trigger: RecoveryTriggerV1) -> str:
    return f"recovery:trigger:{trigger.source_kind}:{trigger.source_id}:{trigger.source_fingerprint}"


def _semantic_hash(request: dict) -> str:
    """Normalized proposal identity: registry, calls, checks and retry; explanatory prose excluded."""
    return _json_hash({
        "registry_version": request.get("registry_version"),
        "calls": [{key: call.get(key) for key in ("tool", "tool_version", "arguments")} for call in request.get("calls", [])],
        "verification": [{key: check.get(key) for key in ("verifier", "verifier_version", "arguments")}
                         for check in request.get("verification", [])],
        "retry": (request.get("retry") or {}).get("target"),
    })


def _delegations(request: dict) -> set[tuple]:
    """Agent + sealed scope/target identity of each delegated remediation. Model prose (objective/result)
    and the time-based deadline/budget are excluded, so rewording cannot repeat a failed delegation."""
    return {(arguments.get("agent_id"), arguments.get("work_function"), _json_hash(arguments.get("actual_execution_scope")),
             _json_hash(arguments.get("retry_target")))
            for call in request.get("calls", []) if call.get("tool") == _DELEGATION_TOOL
            for arguments in [call.get("arguments") or {}]}


def _safe_text(value: object, limit: int = 1_000) -> str:
    return _redact_secrets(str(value))[:limit]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class OrchestrationBlockerRecoveryService:
    def __init__(self, *, analyzer: RecoveryAnalyzer | None = None, registry: RecoveryToolRegistry | None = None) -> None:
        self.analyzer = analyzer or RecoveryAnalyzer()
        self.registry = registry or RecoveryToolRegistry()
        self.orchestration = OrchestrationService()
        self.authority = OrchestrationAuthorityDecisionService()

    # ---- trigger ---------------------------------------------------------------------------------

    async def create_trigger(self, db: AsyncSession, trigger: RecoveryTriggerV1) -> OrchestrationAction:
        """Idempotent on (run_id, source_kind, source_id, source_fingerprint)."""
        run = await db.get(OrchestrationRun, trigger.run_id)
        if run is None or run.goal_id != trigger.goal_id:
            raise HTTPException(status_code=404, detail="Orchestration run not found")
        if trigger.source_kind == "failed_action":
            source = await db.get(OrchestrationAction, trigger.source_id)
            if source is not None and (source.action_type.startswith("recovery_")
                                       or source.idempotency_key.startswith("recovery:")):
                # Recovery's own failures are evidence inside their lineage, never a recursive coordinator.
                raise ValueError("recovery actions are handled inside their lineage")
        return await self.orchestration.reserve_action(
            db, run.id, _trigger_key(trigger), "recovery_trigger", trigger.model_dump(mode="json"),
        )

    # ---- diagnosis -------------------------------------------------------------------------------

    async def diagnose(self, db: AsyncSession, trigger_action_id: UUID) -> OrchestrationAction:
        trigger_action = await db.get(OrchestrationAction, trigger_action_id, populate_existing=True)
        if trigger_action is None or trigger_action.action_type != "recovery_trigger":
            raise HTTPException(status_code=404, detail="Recovery trigger not found")
        trigger = RecoveryTriggerV1.model_validate(trigger_action.request)
        require_baseline_available(trigger.goal_id)
        if trigger_action.status != "reserved":
            return await self._diagnosed_result(db, trigger_action)
        run = await db.get(OrchestrationRun, trigger.run_id)
        goal = await db.get(OrchestrationGoal, trigger.goal_id)
        if run is None or goal is None or run.goal_id != goal.id or run.status not in ACTIVE_RUN_STATUSES:
            raise HTTPException(status_code=409, detail="Recovery run is not active")
        contract = dict(trigger_action.dispatch_contract or {})
        if await self._blocker_gone(db, run, trigger, trigger_action):
            # The source no longer blocks anything: close the trigger without a card or warning.
            async with self.orchestration._lock_goal_for_baseline_transition(db, goal.id):
                await self._require_run_current(db, goal.id, run.id)
                if await self._claim(db, trigger_action):
                    trigger_action.dispatch_contract = {**contract, "closed": "blocker_gone"}
                    await self.orchestration._mark_action_completed(
                        db, trigger_action, "orchestration_action", trigger_action.id)
            return trigger_action

        queries: list[dict] = []
        raw_outputs: list[str] = []
        escalation = await self._escalation(db, run, trigger, trigger_action, contract)
        if escalation is not None:
            snapshot = {"escalation": escalation.model_dump(mode="json")}
        else:
            snapshot = await RecoveryContextBuilder().build(db, trigger, registry_manifest=self.registry.manifest())
            if contract.get("evidence"):
                snapshot["lineage_evidence"] = _bounded({"items": contract["evidence"]}, RECOVERY_EVIDENCE_MAX_BYTES)
        evidence = RecoveryEvidenceService()

        async def recorded_query(query):
            record = {"query": query.model_dump(mode="json")}
            queries.append(record)
            try:
                result = await evidence.query(db, trigger, query)
            except Exception as exc:
                record["error"] = _redact_secrets(str(getattr(exc, "detail", None) or exc))[:1_000]
                raise
            record["result"] = result.model_dump(mode="json")
            return result

        def exchange(assessment=None, rejection: str | None = None) -> OrchestrationDecision:
            row = OrchestrationDecision(
                run_id=run.id, decision_type="recovery",
                input_snapshot={"context": snapshot, "evidence_queries": queries,
                                "trigger_action_id": str(trigger_action.id)},
                llm_output=list(raw_outputs) or None,
                parsed_decision=assessment.model_dump(mode="json") if assessment is not None else {},
                validator_status="rejected" if rejection else "accepted", rejection_reason=rejection,
            )
            db.add(row)
            return row

        async def rejected(assessment, code: str) -> RecoveryDiagnosisFailed:
            row = exchange(assessment, rejection=code)
            await db.flush()
            return RecoveryDiagnosisFailed(trigger_action.id, row.id, code)

        if escalation is not None:
            assessment = escalation
        else:
            # The LLM call runs without any lock; the claim below decides which result persists.
            try:
                assessment = await self.analyzer.analyze(
                    snapshot, project_id=goal.project_id, evidence_query=recorded_query, on_raw=raw_outputs.append,
                )
            except Exception as exc:
                await self._require_run_current(db, goal.id, run.id)
                # F3: a provider/analyzer outage is not a refusal; the tick backs off on this stamp instead.
                trigger_action.dispatch_contract = {**contract, "last_diagnosis_error_at": datetime.now(timezone.utc).isoformat()}
                raise await rejected(None, _redact_secrets(f"{ANALYSIS_FAILED}: {exc}")[:4_000]) from None
        # A reset during the provider call discards this result: nothing is written for the old run.
        await self._require_run_current(db, goal.id, run.id)

        async with self.orchestration._lock_goal_for_baseline_transition(db, goal.id):
            await self._require_run_current(db, goal.id, run.id)
            trigger_action = await db.get(OrchestrationAction, trigger_action.id, populate_existing=True)
            if trigger_action is None:
                raise HTTPException(status_code=409, detail="Baseline operation discarded by reset")
            if trigger_action.status != "reserved":
                return await self._diagnosed_result(db, trigger_action)
            if isinstance(assessment, TransientAssessmentV1):
                # Accepted only when the server proves a relevant live producer for its wake; no warning, no card.
                if not await self._transient_live(db, run, trigger, assessment):
                    raise await rejected(assessment, "transient_without_live_producer")
                if not await self._claim(db, trigger_action):
                    return await self._diagnosed_result(db, trigger_action)
                exchange(assessment)
                await db.flush()
                # ponytail: no wait row; the decision loop keeps running and its next noop is checked live.
                await self.orchestration._mark_action_completed(db, trigger_action, "orchestration_action", trigger_action.id)
                return trigger_action
            # The lineage's blocker warning exists before any proposal is fingerprinted: it mirrors into
            # run.active_blockers (part of the resource fingerprint), so writing it later would stale proposals
            # and defeat same-state dedupe. Deduplicated per lineage root; resolved only by a successful retry.
            await self._ensure_warning(db, goal, run, trigger_action,
                                       _safe_text(f"Blocked, recovery in progress: {trigger.summary}", 4_000))
            proposal = None
            if isinstance(assessment, ProposalDraftV1):
                try:
                    proposal = await self.registry.canonicalize(db, goal, run, trigger, assessment)
                except ValueError as exc:
                    raise await rejected(assessment, _redact_secrets(str(exc))[:4_000]) from None
                refusal = await self._refused_repeat(db, run, contract, proposal)
                if refusal:
                    raise await rejected(assessment, refusal)
            elif not isinstance(assessment, CapabilityGapAssessmentV1):
                raise await rejected(assessment, "unsupported_assessment")
            if not await self._claim(db, trigger_action):  # a concurrent diagnosis won; discard this result
                return await self._diagnosed_result(db, trigger_action)
            row = exchange(assessment)
            await db.flush()
            if proposal is not None:
                result = await self._persist_proposal(db, goal, run, trigger, trigger_action, row, proposal)
            else:
                result = await self._persist_gap(db, goal, run, trigger, trigger_action, row, assessment)
            await self.orchestration._mark_action_completed(db, trigger_action, "orchestration_action", result.id)
            return result

    async def _blocker_gone(self, db, run, trigger: RecoveryTriggerV1, trigger_action: OrchestrationAction) -> bool:
        """An eventless noop whose loop has since done real work, or an observed task failure whose task is no
        longer failed/blocked (e.g. the owner re-ran it), needs no diagnosis."""
        if trigger.source_kind == "eventless_noop":
            return await self._decision_loop_progressed(db, run.id, trigger_action.created_at)
        if trigger.source_kind != "failed_action":
            return False
        source = await db.get(OrchestrationAction, trigger.source_id)
        if (source is None or source.action_type != "failure_observed" or source.target_type != "task"
                or not (source.request or {}).get("closes_when_task_healthy", True)):
            return False
        task = await db.get(Task, source.target_id, populate_existing=True)
        return task is not None and task.status not in {"failed", "blocked"}

    @staticmethod
    async def _claim(db: AsyncSession, trigger_action: OrchestrationAction) -> bool:
        claimed = await db.execute(
            update(OrchestrationAction)
            .where(OrchestrationAction.id == trigger_action.id, OrchestrationAction.status == "reserved")
            .values(status="completed")
            .execution_options(synchronize_session="fetch")
        )
        if claimed.rowcount != 1:
            await db.refresh(trigger_action)
            return False
        return True

    async def _transient_live(self, db, run, trigger: RecoveryTriggerV1, assessment: TransientAssessmentV1) -> bool:
        """The transient wake must pass the same wake-source check as a noop, against current state. The
        blocked source is the failed action's target task when there is one, otherwise the run itself."""
        from huddleroom.services.orchestration_wake_source import OrchestrationWakeSourceService, SourceRef

        source = SourceRef("run", run.id)
        if trigger.source_kind == "failed_action":
            action = await db.get(OrchestrationAction, trigger.source_id)
            if action is not None and action.target_type == "task" and action.target_id is not None:
                source = SourceRef("task", action.target_id)
        verdict = await OrchestrationWakeSourceService().evaluate(
            db, run, blocked_source=source, wake_when=assessment.wake.model_dump(mode="json"),
            live_work_refs=await self.orchestration.live_work_refs(db, run, source),
        )
        return verdict.verdict == "live"

    async def _require_run_current(self, db: AsyncSession, goal_id: UUID, run_id: UUID) -> None:
        """Reset fence: no reservation in progress and the original run still exists and is active."""
        if await self._run_status(db, goal_id, run_id) not in ACTIVE_RUN_STATUSES:
            raise _RunFenced(status_code=409, detail="Recovery run is not active")

    @staticmethod
    async def _run_status(db: AsyncSession, goal_id: UUID, run_id: UUID) -> str:
        """The run's persisted status; 409 while a reset is reserved or after it removed the run."""
        try:
            require_baseline_available(goal_id)
        except HTTPException as exc:
            raise _RunFenced(status_code=exc.status_code, detail=exc.detail) from None
        status = await db.scalar(select(OrchestrationRun.status).where(OrchestrationRun.id == run_id))
        if status is None:
            raise _RunFenced(status_code=409, detail="Baseline operation discarded by reset")
        return status

    @staticmethod
    async def _diagnosed_result(db: AsyncSession, trigger_action: OrchestrationAction) -> OrchestrationAction:
        existing = await db.get(OrchestrationAction, trigger_action.target_id) if trigger_action.target_id else None
        if existing is None:
            raise HTTPException(status_code=409, detail="Recovery trigger is not awaiting diagnosis")
        return existing

    @staticmethod
    async def _lineage_rows(db: AsyncSession, run_id: UUID, lineage_id: str | None) -> list[OrchestrationAction]:
        if not lineage_id:
            return []
        rows = (await db.scalars(select(OrchestrationAction).where(
            OrchestrationAction.run_id == run_id, OrchestrationAction.action_type == "recovery_proposal",
        ))).all()
        return sorted((row for row in rows if (row.dispatch_contract or {}).get("lineage_id") == lineage_id),
                      key=lambda row: (row.dispatch_contract or {}).get("attempt") or 0)

    async def _refused_repeat(self, db, run, contract: dict, proposal: RecoveryProposalV1) -> str | None:
        """Semantic dedupe: same blocker + state + proposal, or a failed delegation's agent/scope again."""
        request = proposal.model_dump(mode="json")
        semantic, delegations = _semantic_hash(request), _delegations(request)
        for prior in await self._lineage_rows(db, run.id, contract.get("lineage_id")):
            prior_request = prior.request or {}
            if not prior_request.get("calls"):
                continue
            if (prior_request.get("resource_fingerprint") == proposal.resource_fingerprint
                    and _semantic_hash(prior_request) == semantic):
                return "duplicate_proposal_on_unchanged_state"
            if prior.status == "failed" and delegations & _delegations(prior_request):
                return "repeat_failed_delegation_same_agent_scope"
        return None

    async def _escalation(self, db, run, trigger: RecoveryTriggerV1, trigger_action: OrchestrationAction,
                          contract: dict) -> CapabilityGapAssessmentV1 | None:
        """After the attempt cap, one actionable human decision replaces any further proposal.

        Two budgets: failed approved attempts in the lineage, and refused diagnoses of this one trigger
        (duplicate/repeat/transient/invalid), so a refused trigger cannot cost an LLM call on every tick."""
        refusals = [row for row in (await db.scalars(select(OrchestrationDecision).where(
            OrchestrationDecision.run_id == run.id, OrchestrationDecision.decision_type == "recovery",
            OrchestrationDecision.validator_status == "rejected",
        ))).all() if (row.input_snapshot or {}).get("trigger_action_id") == str(trigger_action.id)
            # Outages and baseline drift are not refusals of the diagnosis (F3).
            and not (row.rejection_reason or "").startswith(_NOT_A_REFUSAL)]
        if len(refusals) >= MAX_RECOVERY_ATTEMPTS_WITHOUT_PROGRESS:
            return self._escalation_gap(trigger, f"{len(refusals)} diagnoses of this blocker were refused", [
                {"label": "A recovery diagnosis was refused.",
                 "value": {"code": _safe_text(row.rejection_reason or "rejected", 200)},
                 "source_kind": "decision", "source_id": str(row.id)} for row in refusals])
        reset_after = contract.get("budget_reset_after_attempt") or 0
        failed = [row for row in await self._lineage_rows(db, run.id, contract.get("lineage_id"))
                  # A rejection is a human response and a stale proposal is an external state change; neither
                  # is a failed attempt without progress.
                  if row.status == "failed" and row.error not in {"rejected_by_human", "state_changed", ADMISSION_REJECTED}
                  and (row.request or {}).get("calls")
                  and ((row.dispatch_contract or {}).get("attempt") or 0) > reset_after]
        if len(failed) < MAX_RECOVERY_ATTEMPTS_WITHOUT_PROGRESS:
            return None
        return self._escalation_gap(trigger, f"{len(failed)} materially different approved fixes failed", [
            {"label": f"Recovery attempt {(row.dispatch_contract or {}).get('attempt')} failed.",
             "value": {"status": row.status, "error": _safe_text(row.error or "failed", 200)},
             "source_kind": "action", "source_id": str(row.id)} for row in failed])

    @staticmethod
    def _escalation_gap(trigger: RecoveryTriggerV1, what: str, evidence: list[dict]) -> CapabilityGapAssessmentV1:
        return CapabilityGapAssessmentV1.model_validate({
            "kind": "capability_gap",
            "diagnosis": (f"{what} for blocker {trigger.blocker_code} without progress; "
                          "no further fix is proposed automatically."),
            "evidence": evidence[-32:],
            "missing_capability": "An executable remedy that the registered recovery tools have not achieved.",
            "required_human_action": (
                "Review the failed recovery attempts, then supply the missing input, access, deployment or decision "
                "and mark it done. No executable remedy is known with the current tools."),
            "wake_after": {"kind": "human_decision", "decision": "Missing input provided"},
        })

    async def _persist_proposal(self, db, goal, run, trigger, trigger_action, exchange,
                                proposal: RecoveryProposalV1) -> OrchestrationAction:
        lineage = trigger_action.dispatch_contract or {}
        if lineage.get("lineage_id"):
            proposal = proposal.model_copy(update={"lineage_id": UUID(lineage["lineage_id"]), "attempt": lineage["attempt"]})
        request = proposal.model_dump(mode="json")
        proposal_hash = _json_hash(request)
        action = await self.orchestration.reserve_action(
            db, run.id, f"recovery:{proposal.proposal_id}:proposal", "recovery_proposal", request, decision_id=exchange.id,
        )
        action.dispatch_contract = {
            "proposal_hash": proposal_hash, "resource_fingerprint": proposal.resource_fingerprint,
            "lineage_id": str(proposal.lineage_id), "attempt": proposal.attempt,
            "registry_version": proposal.registry_version, "trigger_action_id": str(trigger_action.id),
            "steering": await _steering_seal(db, goal, run),
        }
        response = RecoveryProposalResponse(
            kind="recovery_proposal_v1", proposal_id=proposal.proposal_id, version=proposal.attempt,
            status="pending_approval", problem=proposal.problem, general_updates=proposal.general_updates,
            after_approval=AFTER_APPROVAL, global_effect=proposal.global_effect,
            exact_details=RecoveryConfigurationService._disclosure_value({
                "proposal_hash": proposal_hash, "resource_fingerprint": proposal.resource_fingerprint,
                "registry_version": proposal.registry_version, "calls": request["calls"],
                "verification": request["verification"], "retry": request["retry"],
            }),
            evidence=proposal.evidence, **self._source_links(trigger),
        )
        await self.authority.create_recovery_approval(
            db, goal.id, run_id=run.id, proposal_action_id=action.id, proposal_id=proposal.proposal_id,
            proposal_hash=proposal_hash, question=proposal.problem, context=response.model_dump_json(),
        )
        await db.flush()
        return action

    async def _persist_gap(self, db, goal, run, trigger, trigger_action, exchange,
                           gap: CapabilityGapAssessmentV1) -> OrchestrationAction:
        gap_id = uuid.uuid4()
        lineage = trigger_action.dispatch_contract or {}
        request = {
            "schema_version": 1, "kind": "capability_gap", "gap_id": str(gap_id),
            "source": {"kind": trigger.source_kind, "id": str(trigger.source_id),
                       "blocker_code": trigger.blocker_code, "fingerprint": trigger.source_fingerprint},
            "assessment": gap.model_dump(mode="json"),
        }
        action = await self.orchestration.reserve_action(
            db, run.id, f"recovery:{gap_id}:capability_gap", "recovery_proposal", request, decision_id=exchange.id,
        )
        action.dispatch_contract = {
            "trigger_action_id": str(trigger_action.id),
            "lineage_id": lineage.get("lineage_id") or str(uuid.uuid4()), "attempt": lineage.get("attempt") or 1,
        }
        # Never an approve_fix card: the human supplies the named capability, then recovery re-diagnoses.
        await self.authority.create_pending(
            db, goal.id, decision_key=f"{_GAP_KEY_PREFIX}{gap_id}", title="Capability needed",
            question=gap.required_human_action, authority="human",
            options=[{"key": "capability_provided", "label": "Done"}],
            context=gap.model_dump_json(), run_id=run.id, related_action_id=action.id,
        )
        await self._ensure_warning(db, goal, run, trigger_action, gap.missing_capability)
        await db.flush()
        return action

    @staticmethod
    def _source_links(trigger: RecoveryTriggerV1) -> dict:
        if trigger.source_kind == "failed_action":
            return {"source_action_id": trigger.source_id}
        if trigger.source_kind == "reported_blocker":
            return {"source_decision_id": trigger.source_id}
        return {"source_blocker_id": trigger.source_id}

    # ---- answer ----------------------------------------------------------------------------------

    async def reserve_answer(
        self, db: AsyncSession, decision: OrchestrationAuthorityDecision, *,
        selected_option: Literal["approve_fix", "reject"], reason: str | None, user_id: UUID,
    ) -> RecoveryAnswerReservation:
        if selected_option not in ("approve_fix", "reject"):
            raise ValueError("recovery proposals accept only approve_fix or reject")
        if selected_option == "reject" and not (reason and reason.strip()):
            raise ValueError("rejecting a fix proposal requires a reason")
        if not decision.decision_key.startswith(RECOVERY_DECISION_KEY_PREFIX) or decision.related_action_id is None:
            raise ValueError("not a recovery proposal decision")
        require_baseline_available(decision.goal_id)

        async with self.orchestration._lock_goal_for_baseline_transition(db, decision.goal_id):
            require_baseline_available(decision.goal_id)
            current = await db.get(OrchestrationAuthorityDecision, decision.id, populate_existing=True)
            if current is None:  # e.g. removed by a concurrent reset
                raise ValueError("recovery decision no longer exists")
            proposal = await db.get(OrchestrationAction, current.related_action_id, populate_existing=True)
            if proposal is None or proposal.action_type != "recovery_proposal":
                raise ValueError("recovery proposal action is missing")
            if current.status == "answered":
                if current.selected_option != selected_option:
                    raise ValueError("conflicting duplicate recovery answer")
                return await self._reservation(db, current, proposal, selected_option)
            if current.status != "pending":
                raise ValueError(f"cannot answer decision in status {current.status!r}")
            if proposal.status != "reserved":
                raise ValueError("recovery proposal is no longer awaiting approval")

            run = await db.get(OrchestrationRun, proposal.run_id, populate_existing=True)
            goal = await db.get(OrchestrationGoal, current.goal_id, populate_existing=True)
            if selected_option == "reject":
                await self.authority.answer_recovery_decision(
                    db, current, selected_option="reject", reason=reason.strip(), decided_by_user_id=user_id,
                )
                # The required reason is evidence for the next diagnosis; the rejected proposal is never reapplied.
                await self._fail_attempt(db, proposal, run, goal, "rejected_by_human", [EvidenceV1(
                    label="The human rejected the proposed fix.", value={"reason": _safe_text(reason.strip())},
                    source_kind="decision", source_id=current.id)])
                return RecoveryAnswerReservation(current.id, proposal.id, None)

            parsed = await self._current_proposal(db, current, proposal, goal, run)
            if parsed is None:
                # Flushed, not committed: the caller owns the transaction, catches this, commits, then 409s.
                await self._fail_stale(db, current, proposal, run, goal)
                raise RecoveryProposalStale(proposal.id, current.id)

            await self.authority.answer_recovery_decision(
                db, current, selected_option="approve_fix", reason=reason, decided_by_user_id=user_id,
            )
            return await self._reservation(db, current, proposal, "approve_fix", parsed)

    async def _current_proposal(self, db, decision, proposal, goal, run) -> RecoveryProposalV1 | None:
        """Recompute hash, run identity, registry version and resource fingerprint; None means stale."""
        contract = proposal.dispatch_contract or {}
        proposal_hash = _json_hash(proposal.request)
        if (proposal.status != "reserved" or run is None or goal is None or run.goal_id != goal.id
                or run.status not in ACTIVE_RUN_STATUSES or decision.run_id != run.id
                or contract.get("proposal_hash") != proposal_hash or not decision.decision_key.endswith(proposal_hash)):
            return None
        try:
            parsed = RecoveryProposalV1.model_validate(proposal.request)
            context = RecoveryProposalResponse.model_validate_json(decision.context or "")
            if (context.proposal_id != parsed.proposal_id
                    or parsed.registry_version != registry_module.REGISTRY_VERSION
                    or contract.get("resource_fingerprint") != parsed.resource_fingerprint):
                return None
            resources = await self.registry.capture_resources(db, goal, run, parsed)
        except ValueError:
            return None
        return parsed if resources.fingerprint == parsed.resource_fingerprint else None

    async def _fail_stale(self, db, decision, proposal, run, goal) -> None:
        await self.authority.cancel_recovery_decision(db, decision, reason="state_changed")
        await self._after_failure(db, proposal, run, goal, "state_changed", [])

    async def _reservation(self, db, decision, proposal, selected_option,
                           parsed: RecoveryProposalV1 | None = None) -> RecoveryAnswerReservation:
        if selected_option == "reject":
            return RecoveryAnswerReservation(decision.id, proposal.id, None)
        parsed = parsed or RecoveryProposalV1.model_validate(proposal.request)
        wrapper = await self._reserve_apply_wrapper(db, decision, proposal, parsed.calls[0])
        return RecoveryAnswerReservation(decision.id, proposal.id, wrapper.id)

    async def _reserve_apply_wrapper(self, db, decision, proposal, call) -> OrchestrationAction:
        proposal_hash = proposal.dispatch_contract["proposal_hash"]
        wrapper = await self.orchestration.reserve_action(
            db, proposal.run_id, f"recovery:{proposal.request['proposal_id']}:apply:{call.call_id}", "recovery_apply",
            {"proposal_action_id": str(proposal.id), "proposal_hash": proposal_hash, "call_id": call.call_id},
        )
        if not wrapper.dispatch_contract:
            wrapper.dispatch_contract = {
                "authority_decision_id": str(decision.id), "proposal_hash": proposal_hash,
                "canonical_request_hash": _json_hash(call.arguments),
            }
            await db.flush()
        return wrapper

    async def answer_capability_gap(
        self, db: AsyncSession, decision: OrchestrationAuthorityDecision, *, user_id: UUID,
    ) -> OrchestrationAction:
        """The human supplied the named capability: record it and re-diagnose with a fresh attempt budget."""
        if not decision.decision_key.startswith(_GAP_KEY_PREFIX) or decision.related_action_id is None:
            raise ValueError("not a recovery capability decision")
        require_baseline_available(decision.goal_id)
        async with self.orchestration._lock_goal_for_baseline_transition(db, decision.goal_id):
            current = await db.get(OrchestrationAuthorityDecision, decision.id, populate_existing=True)
            if current is None:
                raise ValueError("recovery decision no longer exists")
            if current.run_id is not None:
                await self._require_run_current(db, current.goal_id, current.run_id)
            if current.status == "pending":
                await self.authority.answer_decision(
                    db, current, selected_option="capability_provided", decided_by_user_id=user_id)
            gap = await db.get(OrchestrationAction, current.related_action_id)
            if gap is None:
                raise HTTPException(status_code=404, detail="Recovery proposal not found")
            return await self.rediagnose(db, gap, EvidenceV1(
                label="The human reported that the requested capability is now available.",
                value={"selected_option": "capability_provided"}, source_kind="decision", source_id=current.id))

    # ---- lifecycle -------------------------------------------------------------------------------

    async def _context(self, db: AsyncSession, proposal_action_id: UUID) -> SimpleNamespace:
        proposal = await db.get(OrchestrationAction, proposal_action_id, populate_existing=True)
        if proposal is None or proposal.action_type != "recovery_proposal":
            raise HTTPException(status_code=404, detail="Recovery proposal not found")
        run = await db.get(OrchestrationRun, proposal.run_id, populate_existing=True)
        goal = await db.get(OrchestrationGoal, run.goal_id) if run is not None else None
        if run is None or goal is None:
            raise HTTPException(status_code=404, detail="Recovery proposal not found")
        require_baseline_available(goal.id)
        decision = await self._decision(db, proposal)
        parsed = RecoveryProposalV1.model_validate(proposal.request) if (proposal.request or {}).get("calls") else None
        approved = (decision is not None and decision.status == "answered" and decision.selected_option == "approve_fix")
        return SimpleNamespace(proposal=proposal, run=run, goal=goal, decision=decision, parsed=parsed,
                               approved=approved, pid=(proposal.request or {}).get("proposal_id"))

    @staticmethod
    async def _decision(db: AsyncSession, proposal: OrchestrationAction) -> OrchestrationAuthorityDecision | None:
        return (await db.scalars(select(OrchestrationAuthorityDecision).where(
            OrchestrationAuthorityDecision.related_action_id == proposal.id,
            OrchestrationAuthorityDecision.decision_key.startswith(RECOVERY_DECISION_KEY_PREFIX),
        ).order_by(OrchestrationAuthorityDecision.created_at.desc())
            .execution_options(populate_existing=True))).first()

    async def _wrapper(self, db, ctx, key: str) -> OrchestrationAction | None:
        row = await self.orchestration._existing_action_for_key(db, ctx.run.id, key)
        if row is not None:
            await db.refresh(row)
        return row

    async def _next_call(self, db, ctx) -> SimpleNamespace:
        """The first unfinished sealed call, in declared order, or why there is none."""
        for call in ctx.parsed.calls:
            wrapper = await self._wrapper(db, ctx, f"recovery:{ctx.pid}:apply:{call.call_id}")
            if wrapper is None or wrapper.status == "reserved":
                return SimpleNamespace(kind="call", call=call, wrapper=wrapper)
            if wrapper.status == "failed":
                return SimpleNamespace(kind="failed")
            receipt = ToolReceiptV1.model_validate(wrapper.dispatch_contract["tool_receipt_v1"])
            if receipt.status == "waiting":
                outcome = await self._delegation_outcome(db, ctx, wrapper, receipt)
                if outcome == "waiting":
                    return SimpleNamespace(kind="waiting")
                if outcome != "done":
                    return SimpleNamespace(kind="delegation_failed", error=outcome, wrapper=wrapper, receipt=receipt)
        return SimpleNamespace(kind="done")

    async def _delegation_outcome(self, db, ctx, wrapper, receipt: ToolReceiptV1) -> str:
        """Task status is the truth: done, failed/cancelled (failure evidence), expired, or still waiting."""
        task_ids = [UUID(ref["id"]) for ref in receipt.target_refs if ref.get("kind") == "task"]
        task = await db.get(Task, task_ids[0], populate_existing=True) if task_ids else None
        deadline = (receipt.observed_post_state or {}).get("deadline")
        if task is None:
            outcome = "delegated_task_missing"
        elif task.status == "done":
            outcome = "done"
        elif task.status in _FAILED_TASK_STATUSES:
            outcome = f"delegated_task_{task.status}"
        elif deadline and datetime.fromisoformat(deadline) <= _utcnow():
            outcome = "remediation_deadline_passed"
        else:
            await self._ensure_waits(db, ctx, wrapper, receipt, task)
            return "waiting"
        await self._close_waits(db, ctx.run.id, str(wrapper.id))
        return outcome

    async def _ensure_waits(self, db, ctx, wrapper, receipt: ToolReceiptV1, task: Task) -> None:
        """One open wait per registered task/session event; the deadline is the due recheck."""
        deadline = (receipt.observed_post_state or {}).get("deadline")
        due = datetime.fromisoformat(deadline) if deadline else _utcnow()
        existing = {row.wait_key for row in (await db.scalars(select(OrchestrationWait).where(
            OrchestrationWait.run_id == ctx.run.id, OrchestrationWait.status == "open"))).all()}
        for event in receipt.wait.events:
            key = f"{wrapper.idempotency_key}:wait:{event['event_type']}"
            if key in existing:
                continue
            db.add(OrchestrationWait(
                run_id=ctx.run.id, wait_key=key, owner={"type": _WAIT_OWNER, "id": str(wrapper.id)},
                awaited_event={"event_type": event["event_type"], "matcher": {"task_id": str(task.id)}},
                due_recheck_at=due,
                fallback={"action_type": "attention", "expected_result": receipt.wait.expected_result,
                          "recheck_seconds": max(1, int((due - _utcnow()).total_seconds()))},
            ))
        await db.flush()

    @staticmethod
    async def _close_waits(db, run_id: UUID, owner_ids: str | set[str]) -> None:
        owner_ids = {owner_ids} if isinstance(owner_ids, str) else owner_ids
        for row in (await db.scalars(select(OrchestrationWait).where(
                OrchestrationWait.run_id == run_id, OrchestrationWait.status == "open"))).all():
            if (row.owner or {}).get("type") == _WAIT_OWNER and (row.owner or {}).get("id") in owner_ids:
                row.status, row.cleared_at = "cleared", _utcnow()
        await db.flush()

    @staticmethod
    def _resource_hashes(snapshot) -> dict[str, str]:
        return {key: _json_hash(value) for key, value in snapshot.values.items()}

    async def _state_current(self, db, ctx, call) -> bool:
        """First effect: the approved fingerprint. Later calls: the rolling post-state of the previous call,
        so resources changed by completed calls match their receipts and untouched resources still match."""
        try:
            resources = await self.registry.capture_resources(db, ctx.goal, ctx.run, ctx.parsed)
        except ValueError:
            return False
        index = [item.call_id for item in ctx.parsed.calls].index(call.call_id)
        if index == 0:
            return resources.fingerprint == ctx.parsed.resource_fingerprint
        previous = await self._wrapper(db, ctx, f"recovery:{ctx.pid}:apply:{ctx.parsed.calls[index - 1].call_id}")
        posted = ((previous.dispatch_contract or {}).get("post_resources") if previous is not None else None)
        return posted is not None and posted == self._resource_hashes(resources)

    async def _record_post_state(self, db, ctx, wrapper) -> None:
        try:
            post = self._resource_hashes(await self.registry.capture_resources(db, ctx.goal, ctx.run, ctx.parsed))
        except ValueError:
            post = None  # fails the next call's rolling check closed
        wrapper.dispatch_contract = {**(wrapper.dispatch_contract or {}), "post_resources": post}
        await db.flush()

    async def _fail_reserved(self, db, action: OrchestrationAction | None, error: str) -> None:
        if action is None:
            return
        await db.refresh(action)
        if action.status == "reserved":
            await self.orchestration.mark_action_failed(db, action, error)
            await db.flush()

    async def execute_next_call(self, db: AsyncSession, proposal_action: OrchestrationAction) -> ToolReceiptV1 | None:
        ctx = await self._context(db, proposal_action.id)
        await self._require_run_current(db, ctx.goal.id, ctx.run.id)
        if ctx.parsed is None or ctx.proposal.status != "reserved" or not ctx.approved:
            return None
        step = await self._next_call(db, ctx)
        if step.kind == "delegation_failed":
            await self._fail_attempt(db, ctx.proposal, ctx.run, ctx.goal, step.error, [step.receipt.evidence[0]])
            return None
        if step.kind != "call":
            return None
        call, wrapper = step.call, step.wrapper
        if call.tool == "team_configuration.apply" and holds_goal_lock():
            return None  # roster-first: cannot run under a caller's goal lock (e.g. a tick); the next pass applies it
        linked = wrapper is not None and (wrapper.dispatch_contract or {}).get("effect_action_id")

        async def fenced_execute() -> ToolReceiptV1 | None:
            """Run, state and steering checks sit immediately before the effect, inside the lock that serializes it."""
            nonlocal wrapper
            await self._require_run_current(db, ctx.goal.id, ctx.run.id)
            if not linked:  # an already-linked effect is reconciled, never re-checked or re-dispatched
                if not await self._state_current(db, ctx, call):
                    return None
                wrapper = wrapper or await self._reserve_apply_wrapper(db, ctx.decision, ctx.proposal, call)
            return await self.registry.execute_call(db, ctx.parsed, call, wrapper, ctx.decision)

        try:
            if call.tool == "team_configuration.apply":
                # Roster-first aggregate: it takes the roster lock, then goal locks, fences itself and owns its commit.
                receipt = await fenced_execute()
            else:
                async with self.orchestration._lock_goal_for_baseline_transition(db, ctx.goal.id):
                    receipt = await fenced_execute()
        except _RunFenced:
            raise
        except (SteeringVersionsChanged, RosterStateChanged):
            receipt = None
        except HTTPException as exc:
            if not self.orchestration._is_genuine_rejection(exc):  # pylint: disable=protected-access
                raise  # waits/5xx keep their state; resume replays through the executor's own idempotency
            await self._fail_reserved(db, wrapper, _safe_text(exc.detail, 4_000))
            # Baseline drift since approval is an external state change, not a failed fix (F3).
            code = ADMISSION_REJECTED if str(exc.detail).startswith(ADMISSION_REJECTED) else "apply_failed"
            await self._fail_attempt(db, ctx.proposal, ctx.run, ctx.goal, code, [])
            return None
        except ValueError as exc:
            await self._fail_reserved(db, wrapper, _safe_text(exc, 4_000))
            await self._fail_attempt(db, ctx.proposal, ctx.run, ctx.goal, "apply_failed", [])
            return None
        if receipt is None:  # drift (resources, steering or roster) after approval: never applied
            await self._fail_reserved(db, wrapper, "state_changed")
            await self._fail_attempt(db, ctx.proposal, ctx.run, ctx.goal, "state_changed", [])
            return None
        # Final-commit fence: a reset reserved during the effect discards it (the caller rolls back).
        await self._require_run_current(db, ctx.goal.id, ctx.run.id)
        if receipt.status == "effect_unknown":
            # Uncertain external effect: both rows fail, nothing is replayed, diagnosis gets the evidence.
            await self._fail_reserved(db, await db.get(OrchestrationAction, receipt.action_id), "effect_unknown")
            await self._fail_reserved(db, wrapper, "effect_unknown")
            await self._fail_attempt(db, ctx.proposal, ctx.run, ctx.goal, "effect_unknown", list(receipt.evidence[:1]))
        elif receipt.status == "failed":
            await self._fail_attempt(db, ctx.proposal, ctx.run, ctx.goal, "apply_failed", list(receipt.evidence[:1]))
        else:
            await self._record_post_state(db, ctx, wrapper)
            if receipt.status == "waiting":
                await self._next_call(db, ctx)  # stores the delegated waits (or settles an already-finished task)
        return receipt

    async def verify_proposal(
        self, db: AsyncSession, proposal_action: OrchestrationAction,
    ) -> tuple[VerificationResultV1, ...]:
        ctx = await self._context(db, proposal_action.id)
        await self._require_run_current(db, ctx.goal.id, ctx.run.id)
        if ctx.parsed is None or not ctx.approved:
            raise ValueError("recovery proposal is not approved")
        stored = await self._verify_wrappers(db, ctx)
        if ctx.proposal.status != "reserved":
            return tuple(VerificationResultV1.model_validate(row.dispatch_contract["verification_result_v1"])
                         for row in stored if "verification_result_v1" in (row.dispatch_contract or {}))
        if (await self._next_call(db, ctx)).kind != "done":
            raise ValueError("recovery calls are not complete")
        results: list[VerificationResultV1] = []
        proposal_hash = ctx.proposal.dispatch_contract["proposal_hash"]
        for check in ctx.parsed.verification:
            key = f"recovery:{ctx.pid}:verify:{check.check_id}"
            wrapper = await self._wrapper(db, ctx, key)
            if wrapper is not None and wrapper.status != "reserved":
                results.append(VerificationResultV1.model_validate(wrapper.dispatch_contract["verification_result_v1"]))
                continue
            result = await self.registry.verify(db, ctx.parsed, check)
            await self._require_run_current(db, ctx.goal.id, ctx.run.id)
            wrapper = wrapper or await self.orchestration.reserve_action(
                db, ctx.run.id, key, "recovery_verify",
                {"proposal_action_id": str(ctx.proposal.id), "proposal_hash": proposal_hash, "check_id": check.check_id},
            )
            wrapper.dispatch_contract = {**(wrapper.dispatch_contract or {}),
                                         "verification_result_v1": result.model_dump(mode="json")}
            results.append(result)
            if result.status == "passed":
                await self.orchestration._mark_action_completed(db, wrapper, "orchestration_action", ctx.proposal.id)
            elif result.status == "failed":
                await self.orchestration.mark_action_failed(db, wrapper, "verification_failed")
                await self._fail_attempt(db, ctx.proposal, ctx.run, ctx.goal, "verification_failed", list(result.evidence[:4]))
                break
            else:
                await db.flush()
                break  # pending: a live producer is awaited, nothing else runs
        return tuple(results)

    async def _verify_wrappers(self, db, ctx) -> list[OrchestrationAction]:
        rows = [await self._wrapper(db, ctx, f"recovery:{ctx.pid}:verify:{check.check_id}")
                for check in ctx.parsed.verification]
        return [row for row in rows if row is not None]

    async def retry_blocked_work(self, db: AsyncSession, proposal_action: OrchestrationAction) -> OrchestrationAction:
        ctx = await self._context(db, proposal_action.id)
        await self._require_run_current(db, ctx.goal.id, ctx.run.id)
        if ctx.parsed is None or not ctx.approved:
            raise ValueError("recovery proposal is not approved")
        key = f"recovery:{ctx.pid}:retry:1"
        wrapper = await self._wrapper(db, ctx, key)
        if wrapper is not None and wrapper.status != "reserved":
            return wrapper  # the one approved retry already ran
        if ctx.proposal.status != "reserved":
            raise ValueError("recovery proposal is no longer active")
        verifies = await self._verify_wrappers(db, ctx)
        if len(verifies) != len(ctx.parsed.verification) or any(row.status != "completed" for row in verifies):
            raise ValueError("the retry runs only after every verification check passed")
        wrapper = wrapper or await self.orchestration.reserve_action(
            db, ctx.run.id, key, "recovery_retry",
            {"proposal_action_id": str(ctx.proposal.id), "proposal_hash": ctx.proposal.dispatch_contract["proposal_hash"],
             "approved_retry_ordinal": 1},
        )
        try:
            outcome, effect, error = await self._run_retry(db, ctx, wrapper)
        except _RunFenced:
            raise
        except (SteeringVersionsChanged, RosterStateChanged):
            outcome, effect, error = "failed", None, "state_changed"
        except ValueError as exc:
            outcome, effect, error = "failed", None, _safe_text(exc)
        except HTTPException as exc:
            if not self.orchestration._is_genuine_rejection(exc):  # pylint: disable=protected-access
                raise  # waits/5xx keep their state; the next resume replays the same effect key
            outcome, effect, error = "failed", None, _safe_text(exc.detail)
        await self._require_run_current(db, ctx.goal.id, ctx.run.id)
        if outcome == "pending":
            return wrapper  # retrying: waits for the loop/process to progress; nothing is resolved yet
        if outcome == "completed":
            await self.orchestration._mark_action_completed(db, wrapper, "orchestration_action", effect.id)
            await self.orchestration._mark_action_completed(db, ctx.proposal, "orchestration_action", wrapper.id)
            await self._resolve_lineage(db, ctx)
        else:
            if outcome == "unknown":
                await self._fail_reserved(db, effect, "effect_unknown")
            await self._fail_reserved(db, wrapper, error)
            code = {"unknown": "effect_unknown"}.get(outcome, "state_changed" if error == "state_changed" else "retry_failed")
            await self._fail_attempt(db, ctx.proposal, ctx.run, ctx.goal, code,
                                     [EvidenceV1(label="The approved retry did not complete.",
                                                 value={"error": _safe_text(error)}, source_kind="action",
                                                 source_id=(effect or wrapper).id)])
        await db.refresh(wrapper)
        return wrapper

    async def complete_retry_on_observed_progress(self, db: AsyncSession, proposal_action: OrchestrationAction) -> bool:
        """An owner that observed the blocked work progress outside the decision loop (continuous discovery consumed
        a valid source output) completes the reserved ``decision_loop`` retry and resolves the lineage, exactly as
        ``retry_blocked_work`` does when the loop progresses. False when there is no such reserved retry."""
        ctx = await self._context(db, proposal_action.id)
        if (ctx.parsed is None or not ctx.approved or ctx.proposal.status != "reserved"
                or ctx.parsed.retry.target.kind != "decision_loop"):
            return False
        wrapper = await self._wrapper(db, ctx, f"recovery:{ctx.pid}:retry:1")
        if wrapper is None or wrapper.status != "reserved":
            return False
        await self._require_run_current(db, ctx.goal.id, ctx.run.id)
        await self.orchestration._mark_action_completed(db, wrapper, "orchestration_action", wrapper.id)
        await self.orchestration._mark_action_completed(db, ctx.proposal, "orchestration_action", wrapper.id)
        await self._resolve_lineage(db, ctx)
        return True

    async def _run_retry(self, db, ctx, wrapper) -> tuple[str, OrchestrationAction | None, str | None]:
        """Run (or reconcile) the one sealed retry; returns (completed|pending|failed|unknown, effect, error)."""
        target = ctx.parsed.retry.target
        contract = wrapper.dispatch_contract or {}
        if target.kind in {"decision_loop", "baseline_process"}:
            # In-wrapper retry: the loop/process re-runs on its own tick (calls resume). The retry stays
            # reserved ("retrying") and completes only once that progress is observed.
            wrapper.dispatch_contract = {**contract, "authority_decision_id": str(ctx.decision.id),
                                         "effect_action_id": str(wrapper.id), "effect_idempotency_key": wrapper.idempotency_key}
            if await self._loop_progressed(db, ctx, target, wrapper):
                return "completed", wrapper, None
            # No wait row: an open wait would make reconcile_local report "waiting" and park the very
            # loop/process this retry waits on. The tick polls resume instead.
            return "pending", wrapper, None
        if contract.get("effect_action_id"):
            effect = await db.get(OrchestrationAction, UUID(contract["effect_action_id"]))
        else:
            action_type, request = await self._retry_request(db, ctx, target)
            dispatcher = OrchestrationDecisionDispatcher(self.orchestration)
            decision = (await db.get(OrchestrationDecision, UUID(contract["retry_decision_id"]))
                        if contract.get("retry_decision_id") else None)
            if decision is None:  # one decision row per retry, reused on replay
                decision = OrchestrationDecision(
                    run_id=ctx.run.id, decision_type="recovery_retry", validator_status="accepted", input_snapshot={},
                    parsed_decision={**request, "action_type": action_type, "applies_decision_id": str(ctx.decision.id)},
                )
                db.add(decision)
                await db.flush()
            wrapper.dispatch_contract = {**contract, "authority_decision_id": str(ctx.decision.id),
                                         "proposal_hash": ctx.proposal.dispatch_contract["proposal_hash"],
                                         "canonical_request_hash": dispatcher._canonical_request_hash(request),  # pylint: disable=protected-access
                                         "retry_decision_id": str(decision.id)}
            await db.flush()
            context = RecoveryRetryDispatchContext(purpose="retry", proposal_id=ctx.parsed.proposal_id,
                                                   approved_retry_ordinal=1, authority_decision_id=ctx.decision.id)
            async with self.orchestration._lock_goal_for_baseline_transition(db, ctx.goal.id):
                # The decision carries no steering snapshot (it would re-key the effect), so the retry is fenced
                # here against the run identity and the steering sealed at proposal time, inside the lock.
                await self._require_run_current(db, ctx.goal.id, ctx.run.id)
                await self.registry._assert_sealed_steering(db, ctx.run, ctx.decision)  # pylint: disable=protected-access
                effect = await dispatcher.dispatch(db, ctx.run, decision, recovery_context=context)
            if effect is None:
                return "failed", None, decision.rejection_reason or "recovery_retry_rejected"
        if effect is None:
            return "failed", None, "recovery_effect_link_mismatch"
        await db.refresh(effect)
        if effect.status == "completed":
            if target.kind == "action":
                source = await db.get(OrchestrationAction, target.source_action_id)
                if source is not None and source.action_type == "failure_observed":
                    await self.orchestration.link_recovery_retry_effect(db, source, effect)
            return "completed", effect, None
        if effect.status == "failed":
            return "failed", effect, effect.error or "recovery_retry_failed"
        return "unknown", effect, "effect_unknown"

    async def _loop_progressed(self, db, ctx, target, wrapper) -> bool:
        """baseline_process: the named current process completed. decision_loop: the ordinary loop completed a
        non-recovery, work-performing action after the retry was reserved."""
        if target.kind == "baseline_process":
            current = await OrchestrationProcessService().get_current(db, ctx.goal.id, target.process_type)
            return current is not None and current.status == "completed"
        return await self._decision_loop_progressed(db, ctx.run.id, wrapper.created_at)

    @staticmethod
    async def _decision_loop_progressed(db, run_id: UUID, since: datetime) -> bool:
        # Decision-loop provenance only: a completed work action linked to an accepted, non-recovery
        # orchestrator decision. Supervision/wait bookkeeping and operator/API actions (no decision) never count.
        rows = (await db.execute(select(OrchestrationAction, OrchestrationDecision).join(
            OrchestrationDecision, OrchestrationDecision.id == OrchestrationAction.decision_id,
        ).where(
            OrchestrationAction.run_id == run_id, OrchestrationAction.status == "completed",
            OrchestrationAction.created_at > since,
            OrchestrationDecision.run_id == run_id, OrchestrationDecision.validator_status == "accepted",
        ))).all()
        return any(not decision.decision_type.startswith("recovery") and not action.action_type.startswith("recovery_")
                   and not action.idempotency_key.startswith("recovery:")
                   and f"run:{run_id}:supervision:" not in action.idempotency_key
                   and ":wait:" not in action.idempotency_key
                   and action.action_type not in _NON_PROGRESS_ACTIONS for action, decision in rows)

    async def _retry_request(self, db, ctx, target) -> tuple[str, dict]:
        if target.kind == "action":
            source = await db.get(OrchestrationAction, target.source_action_id)
            if source is None or source.run_id != ctx.run.id:
                raise ValueError("retry_source_action_missing")
            if source.action_type == "failure_observed":
                # Server-sealed gate-linked creation the legacy path would have run (never model-authored).
                sealed = (source.request or {}).get("retry_action")
                if not isinstance(sealed, dict):
                    raise ValueError("retry_source_action_missing")
                if sealed.get("kind") == "final_summary":
                    action_type = "create_delegation_task"
                    values = await self.orchestration.final_summary_retry_request(db, ctx.goal, ctx.run)
                else:
                    action_type, values = sealed["action_type"], dict(sealed["request"])
            else:
                action_type, values = source.action_type, dict(source.request or {})
        else:  # planning
            author = target.requested_agent_id or await select_plan_author(db, ctx.goal, ctx.run)
            if author is None:
                raise HTTPException(status_code=409, detail="plan_author_unavailable")
            action_type = "request_plan"
            values = {"agent_id": str(author), "work_function": "planning",
                      "scope": goal_objective_with_clarifications(ctx.goal)}
        values["action_type"] = action_type
        return action_type, self.orchestration.canonical_decision_request(action_type, values, run_id=ctx.run.id)

    async def resume(self, db: AsyncSession, proposal_action_id: UUID) -> RecoveryLifecycleStatus:
        """Tick/crash re-entry: drive every ready step and stop at the first wait or terminal state."""
        ctx = await self._context(db, proposal_action_id)
        if ctx.parsed is not None and ctx.proposal.status == "reserved" and ctx.approved:
            await self._require_run_current(db, ctx.goal.id, ctx.run.id)
            for _ in range(len(ctx.parsed.calls) + 1):
                step = await self._next_call(db, ctx)
                if step.kind == "call":
                    receipt = await self.execute_next_call(db, ctx.proposal)
                    if receipt is None or receipt.status in {"failed", "effect_unknown"}:
                        break
                    continue
                if step.kind == "delegation_failed":
                    await self._fail_attempt(db, ctx.proposal, ctx.run, ctx.goal, step.error, [step.receipt.evidence[0]])
                elif step.kind == "done":
                    results = await self.verify_proposal(db, ctx.proposal)
                    if len(results) == len(ctx.parsed.verification) and all(item.status == "passed" for item in results):
                        await self.retry_blocked_work(db, ctx.proposal)
                break
        await db.refresh(ctx.proposal)
        return await self.project_status(db, ctx.proposal)

    async def rediagnose(
        self, db: AsyncSession, proposal_action: OrchestrationAction, evidence: EvidenceV1,
    ) -> OrchestrationAction:
        """Re-enter diagnosis in the same lineage (attempt + 1) with the new evidence."""
        ctx = await self._context(db, proposal_action.id)
        await self._require_run_current(db, ctx.goal.id, ctx.run.id)
        budget_reset = None
        if (ctx.proposal.request or {}).get("kind") == "capability_gap":
            gap_decision = (await db.scalars(select(OrchestrationAuthorityDecision).where(
                OrchestrationAuthorityDecision.related_action_id == ctx.proposal.id,
                OrchestrationAuthorityDecision.decision_key.startswith(_GAP_KEY_PREFIX),
            ).execution_options(populate_existing=True))).first()
            if gap_decision is None or gap_decision.selected_option != "capability_provided":
                raise ValueError("the capability decision has not been answered")
            # A human response resets the diagnosis budget (spec).
            budget_reset = ctx.proposal.dispatch_contract.get("attempt") or 1
            if ctx.proposal.status == "reserved":
                await self.orchestration._mark_action_completed(db, ctx.proposal, "authority_decision", gap_decision.id)
        elif ctx.proposal.status != "failed":
            raise ValueError("only a failed or rejected recovery attempt can be re-diagnosed")
        async with self.orchestration._lock_goal_for_baseline_transition(db, ctx.goal.id):
            await self._require_run_current(db, ctx.goal.id, ctx.run.id)
            return await self._follow_up_trigger(db, ctx.proposal, ctx.run, [evidence], budget_reset=budget_reset)

    # ---- failure, lineage and warnings -------------------------------------------------------------

    async def _fail_attempt(self, db, proposal, run, goal, error: str, evidence: list[EvidenceV1]) -> None:
        """Fail the attempt (never a rollback claim: applied calls stay applied) and re-enter diagnosis."""
        await self._run_status(db, goal.id, run.id)  # reset fence: an old run writes nothing more
        if proposal.status == "reserved":
            await self.orchestration.mark_action_failed(db, proposal, error)
        wrappers = [row for row in (await db.scalars(select(OrchestrationAction).where(
            OrchestrationAction.run_id == run.id, OrchestrationAction.action_type.in_(_WRAPPER_TYPES),
            OrchestrationAction.idempotency_key.startswith(f"recovery:{(proposal.request or {}).get('proposal_id')}:"),
        ))).all() if (row.request or {}).get("proposal_action_id") == str(proposal.id)]
        for row in wrappers:
            if row.status == "reserved":
                await self.orchestration.mark_action_failed(db, row, error)
            effect_id = (row.dispatch_contract or {}).get("effect_action_id")
            if effect_id and effect_id != str(row.id):  # a linked domain effect still reserved fails with it
                await self._fail_reserved(db, await db.get(OrchestrationAction, UUID(effect_id)), error)
        await self._close_waits(db, run.id, {str(row.id) for row in wrappers})
        receipts = [{"call_id": receipt["call_id"], "status": receipt["status"], "wrapper_action_id": receipt["wrapper_action_id"],
                     "action_id": receipt["action_id"]}
                    for row in wrappers if (receipt := (row.dispatch_contract or {}).get("tool_receipt_v1"))]
        summary = EvidenceV1(
            label=f"Recovery attempt {(proposal.dispatch_contract or {}).get('attempt', 1)} failed.",
            value={"error": _safe_text(error, 300), "applied_state": receipts}, source_kind="action", source_id=proposal.id)
        await self._after_failure(db, proposal, run, goal, error, [summary, *evidence])

    async def _after_failure(self, db, proposal, run, goal, error: str, evidence: list[EvidenceV1]) -> None:
        if run is None or goal is None:
            return
        # Persisted status, not the in-memory row: a reset or terminal run never re-diagnoses or warns.
        if await self._run_status(db, goal.id, run.id) not in ACTIVE_RUN_STATUSES:
            return
        trigger_action = await self._trigger_of(db, proposal)
        await self._ensure_warning(db, goal, run, trigger_action,
                                   f"Recovery attempt failed ({_safe_text(error, 200)}); the blocker is still active.")
        await self._follow_up_trigger(db, proposal, run, evidence)
        await db.flush()

    @staticmethod
    async def _trigger_of(db, proposal) -> OrchestrationAction:
        return await db.get(OrchestrationAction, UUID((proposal.dispatch_contract or {})["trigger_action_id"]))

    @staticmethod
    def _root_id(trigger_action: OrchestrationAction) -> str:
        return (trigger_action.dispatch_contract or {}).get("root_trigger_action_id") or str(trigger_action.id)

    async def _follow_up_trigger(self, db, proposal, run, evidence: list[EvidenceV1], *,
                                 budget_reset: int | None = None) -> OrchestrationAction:
        contract = proposal.dispatch_contract or {}
        trigger_action = await self._trigger_of(db, proposal)
        root = self._root_id(trigger_action)
        attempt = contract["attempt"] + 1
        # Bounded key: root trigger + attempt, never a growing suffix chain.
        row = await self.orchestration.reserve_action(
            db, run.id, f"recovery:trigger:{root}:attempt:{attempt}", "recovery_trigger", trigger_action.request,
        )
        if row.status == "reserved":
            current = dict(row.dispatch_contract or {}) or {
                "root_trigger_action_id": root, "lineage_id": contract["lineage_id"], "attempt": attempt,
                "supersedes_proposal_action_id": str(proposal.id),
                "budget_reset_after_attempt": (trigger_action.dispatch_contract or {}).get("budget_reset_after_attempt"),
            }
            if budget_reset is not None:
                current["budget_reset_after_attempt"] = budget_reset
            items = list(current.get("evidence") or [])
            for item in evidence:
                value = item.model_dump(mode="json")
                if value not in items:
                    items.append(value)
            current["evidence"] = items[-_MAX_TRIGGER_EVIDENCE:]
            row.dispatch_contract = current
        await db.flush()
        return row

    async def _ensure_warning(self, db, goal, run, trigger_action, message: str) -> None:
        """One active blocker warning per lineage (deduplicated on its root trigger) blocks an on-track view."""
        if run is None or trigger_action is None or await self._run_status(db, goal.id, run.id) not in ACTIVE_RUN_STATUSES:
            return
        await OrchestrationWarningService().create_warning(
            db, goal.id, warning_type=_WARNING_TYPE, severity="blocker", message=message[:4_000],
            run_id=run.id, related_action_id=UUID(self._root_id(trigger_action)),
        )

    async def _resolve_lineage(self, db, ctx) -> None:
        """The retry removed the blocker: resolve the lineage's warnings and waits with proposal provenance."""
        root = UUID(self._root_id(await self._trigger_of(db, ctx.proposal)))
        warnings = (await db.scalars(select(OrchestrationWarning).where(
            OrchestrationWarning.goal_id == ctx.goal.id, OrchestrationWarning.warning_type == _WARNING_TYPE,
            OrchestrationWarning.related_action_id == root, OrchestrationWarning.active.is_(True),
        ))).all()
        service = OrchestrationWarningService()
        for warning in warnings:
            await service.resolve_warning(
                db, warning, resolved_by="orchestrator:recovery",
                reason=f"Recovery proposal {ctx.pid} verified its checks and its retry completed.")
        wrappers = [row for row in (await db.scalars(select(OrchestrationAction).where(
            OrchestrationAction.run_id == ctx.run.id, OrchestrationAction.action_type.in_(("recovery_apply", "recovery_retry")),
            OrchestrationAction.idempotency_key.startswith(f"recovery:{ctx.pid}:")))).all()]
        await self._close_waits(db, ctx.run.id, {str(row.id) for row in wrappers})

    # ---- projection ------------------------------------------------------------------------------

    async def project_status(self, db: AsyncSession, proposal_action: OrchestrationAction) -> RecoveryLifecycleStatus:
        request = proposal_action.request or {}
        if request.get("kind") == "capability_gap":
            return RecoveryLifecycleStatus.UNSUPPORTED
        if proposal_action.status == "completed":
            return RecoveryLifecycleStatus.RESOLVED
        if proposal_action.status == "failed":
            return (RecoveryLifecycleStatus.REJECTED if proposal_action.error == "rejected_by_human"
                    else RecoveryLifecycleStatus.FAILED)
        decision = await self._decision(db, proposal_action)
        if decision is None or decision.status == "cancelled":
            return RecoveryLifecycleStatus.FAILED
        if decision.status == "pending":
            return RecoveryLifecycleStatus.PENDING_APPROVAL
        if decision.selected_option == "reject":
            return RecoveryLifecycleStatus.REJECTED
        prefix = f"recovery:{request.get('proposal_id')}:"
        wrappers = [row for row in (await db.scalars(select(OrchestrationAction).where(
            OrchestrationAction.run_id == proposal_action.run_id,
            OrchestrationAction.action_type.in_(_WRAPPER_TYPES),
            OrchestrationAction.idempotency_key.startswith(prefix),
        ).execution_options(populate_existing=True))).all() if (row.request or {}).get("proposal_action_id") == str(proposal_action.id)]
        if any(row.status == "failed" for row in wrappers):
            return RecoveryLifecycleStatus.FAILED
        by_type = {kind: [row for row in wrappers if row.action_type == kind] for kind in _WRAPPER_TYPES}
        verifies = by_type["recovery_verify"]
        if by_type["recovery_retry"] or (
                verifies and len(verifies) == len(request.get("verification", []))
                and all(row.status == "completed" for row in verifies)):
            return RecoveryLifecycleStatus.RETRYING  # verified, not resolved until the retry progressed
        if verifies and all(row.status == "completed" for row in by_type["recovery_apply"]):
            return RecoveryLifecycleStatus.VERIFYING
        return RecoveryLifecycleStatus.APPLYING
