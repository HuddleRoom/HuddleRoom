from __future__ import annotations

import hashlib
import json
import logging
import uuid
from typing import Any

from fastapi import HTTPException
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.orchestration import OrchestrationAction, OrchestrationDecision, OrchestrationGoal, OrchestrationRun
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
from huddleroom.schemas.orchestration_recovery import (
    CorrectiveDisposition, RecoveryDispatchContext, RecoveryProposalV1,
)
from huddleroom.services.orchestration_steering import (
    SteeringVersionsChanged,
    OrchestrationSteeringService, active_direction_ids,
    steering_versions_from_snapshot,
)

logger = logging.getLogger(__name__)


def recovery_wrapper_key(context: RecoveryDispatchContext) -> str:
    if context.purpose == "apply":
        return f"recovery:{context.proposal_id}:apply:{context.call_id}"
    return f"recovery:{context.proposal_id}:retry:{context.approved_retry_ordinal}"


def recovery_effect_key(context: RecoveryDispatchContext) -> str:
    return f"{recovery_wrapper_key(context)}:effect"


class OrchestrationDecisionDispatcher:
    """Maps one validated LLM coordination decision to the matching
    execute_*_action on OrchestrationService. Code routes; the LLM only chose
    the action. Exactly one action per decision, recorded idempotently."""

    # Actions that do not perform applied work, so an answered decision is not
    # recorded on them.
    _UNLINKED_ACTION_TYPES = frozenset(
        {"noop", "record_warning", "pause_run", "ask_human", "suggest_agent", "recover_blocker"}
    )

    # Executors whose 4xx failure state (plan revision, replan attention) is read by the runner,
    # so it must stay committed and the error must propagate.
    _OWN_RECOVERY_ACTION_TYPES = frozenset({"accept_plan", "request_roadmap_replan"})

    def __init__(self, service: Any) -> None:
        self._service = service

    @staticmethod
    def action_key(run_id: uuid.UUID, action_type: str, request: dict[str, Any]) -> str:
        """Stable replay identity for a regenerated coordination decision.

        Reasons explain a decision but do not change its requested work.
        """
        fingerprint_request = {key: value for key, value in request.items() if key != "reason"}
        encoded = json.dumps(
            fingerprint_request, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        fingerprint = hashlib.sha256(encoded).hexdigest()[:16]
        return f"run:{run_id}:kind:{action_type}:request:{fingerprint}"

    async def dispatch(
        self,
        db: AsyncSession,
        run: OrchestrationRun,
        decision: OrchestrationDecision,
        *,
        recovery_context: RecoveryDispatchContext | None = None,
    ) -> OrchestrationAction | None:
        if decision.validator_status != "accepted":
            return None
        parsed = decision.parsed_decision or {}
        action_type = parsed.get("action_type")
        if not action_type:
            return None

        executor = self._EXECUTORS.get(action_type)
        if executor is None:
            # Validated but not dispatchable from the runtime loop (e.g. baseline
            # authority actions handled elsewhere). No-op rather than guess.
            return None
        applies_decision_id = None
        if "applies_decision_id" in parsed:
            applies_decision_id, rejection = await self._resolve_applied_decision(
                db, run, parsed["applies_decision_id"],
            )
            if rejection is not None:
                decision.validator_status = "rejected"
                decision.rejection_reason = rejection
                return None
        request = self._service.canonical_decision_request(action_type, parsed, run_id=run.id)
        recovery = await self._recovery_dispatch_context(
            db, run, applies_decision_id, recovery_context, request,
        )
        if recovery is None and recovery_context is not None:
            decision.validator_status = "rejected"
            decision.rejection_reason = "invalid recovery dispatch context"
            return None
        if recovery is False:
            decision.validator_status = "rejected"
            decision.rejection_reason = "invalid recovery dispatch identity"
            return None
        if recovery_context is None:
            # Every corrective fix needs approval: a model-selected corrective action becomes one recovery
            # trigger and nothing is mutated. Only an approved recovery dispatch context may apply it.
            disposition = await self._service.classify_corrective_action(db, run, action_type, request, source="model")
            if disposition == CorrectiveDisposition.REQUIRES_RECOVERY:
                return await self._corrective_trigger(db, run, decision, action_type, request)
        input_snapshot = getattr(decision, "input_snapshot", {}) or {}
        versions = steering_versions_from_snapshot(input_snapshot)
        steering = None
        if versions is not None:
            goal = await db.get(OrchestrationGoal, run.goal_id)
            if goal is None:
                return None
            steering = OrchestrationSteeringService()
            await steering.assert_current_versions(db, goal, run, versions)
            if active_direction_ids(input_snapshot) and action_type in {"pause_run", "ask_human"}:
                # Model-proposed: reject (noop/backstop wait follows) rather than crash the tick.
                decision.validator_status = "rejected"
                decision.rejection_reason = "steering_forbidden_effect: Use the dedicated control"
                return None
        method = getattr(self._service, executor)
        if recovery:
            action_key = recovery_effect_key(recovery_context)
        elif action_type == "request_roadmap_replan":
            action_key = await self._service.roadmap_replan_action_key(db, run)
        else:
            action_key = self.action_key(run.id, action_type, request)
            if action_type == "request_verification":
                action_key += await self._service.verification_attempt_suffix(db, run.id, request.get("gate_id"))
        if not recovery:
            action_key, capped = await self._attempt_key(db, run.id, action_key, decision.id, versions is not None)
            if capped is not None:
                # Rejected too often without progress: a durable failure without its own successful recovery.
                await self._service.create_recovery_trigger(
                    db, run, source_kind="failed_action", source_id=capped.id, blocker_code="action_rejected",
                    summary=f"{action_type} was rejected repeatedly: {capped.error or 'rejected'}",
                    retry_target={"kind": "action", "source_action_id": str(capped.id)},
                    fingerprint_basis={"source": str(capped.id)},
                )
                return capped  # rejected too often: no re-run, no new row; the backstop wait still applies
        if versions is not None:
            action_key, *_ = (
                await self._service._steering_action_fence(db, run.id, action_key, decision.id)
            )
        run_id, decision_id = run.id, decision.id
        # Only model-proposed delegation is ad hoc; internal callers keep the default.
        extra = {"ad_hoc": True} if action_type == "create_delegation_task" else {}
        if extra and recovery and recovery_context.purpose == "apply" and recovery_context.baseline_repair_admission:
            # Trusted only from a validated approve_fix context; the executor re-checks it against current state.
            extra["baseline_repair_admission"] = recovery_context.baseline_repair_admission
        call = lambda: method(  # noqa: E731
            db, run_id=run_id, request=request, idempotency_key=action_key, decision_id=decision_id, **extra,
        )
        if action_type in self._OWN_RECOVERY_ACTION_TYPES:
            if action_type == "request_roadmap_replan" and await self._service._existing_action_for_key(
                db, run_id, action_key
            ) is None:
                try:
                    await self._service._require_plan_author(
                        db, await db.get(OrchestrationGoal, run.goal_id), run, request.get("agent_id")
                    )
                except HTTPException as exc:
                    if not str(exc.detail).startswith("plan_author_"):
                        raise
                    return await self._record_rejection(
                        db, run_id, decision_id, action_type, request, action_key, str(exc.detail),
                    )
            action = await call()
        else:
            kept: Exception | None = None
            marks = self._service._side_effect_marks(db)
            try:
                # Savepoint: a rejected executor must not leave partial work (budget, meetings, graphs).
                async with self._savepoint(db):
                    try:
                        action = await call()
                    except SteeringVersionsChanged as exc:
                        kept = exc  # its failed row must stay
                    except HTTPException as exc:
                        if self._service._is_genuine_rejection(exc):
                            raise
                        kept = exc  # waits/5xx persist state on purpose
            except HTTPException as exc:
                await self._service._after_savepoint_rollback(db, run, marks)
                action = await self._record_rejection(
                    db, run_id, decision_id, action_type, request, action_key, str(exc.detail),
                )
            except Exception:
                await self._service._after_savepoint_rollback(db, run, marks)
                raise
            if kept is not None:
                raise kept
        if (
            applies_decision_id is not None
            and action.status == "completed"
            and action.action_type not in self._UNLINKED_ACTION_TYPES
        ):
            # setdefault: a replayed action keeps the decision it was first linked to.
            contract = dict(action.dispatch_contract or {})
            contract.setdefault("applies_decision_id", str(applies_decision_id))
            action.dispatch_contract = contract
        if recovery:
            self.link_recovery_effect(recovery, recovery_context, request, action)
        if steering is not None and action.status != "failed":
            for request_id in active_direction_ids(input_snapshot):
                await steering.link_result(db, request_id, decision.id, action.id)
        return action

    @classmethod
    def link_recovery_effect(
        cls, wrapper: OrchestrationAction, context: RecoveryDispatchContext,
        request: dict[str, Any], action: OrchestrationAction,
    ) -> None:
        """Durable wrapper-to-effect link of one recovery dispatch."""
        wrapper.dispatch_contract = {
            **(wrapper.dispatch_contract or {}),
            "authority_decision_id": str(context.authority_decision_id),
            "canonical_request_hash": cls._canonical_request_hash(request),
            "effect_action_id": str(action.id),
            "effect_idempotency_key": action.idempotency_key,
        }

    @staticmethod
    def _canonical_request_hash(request: dict[str, Any]) -> str:
        encoded = json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    async def _recovery_dispatch_context(
        self,
        db: AsyncSession,
        run: OrchestrationRun,
        applies_decision_id: uuid.UUID | None,
        context: RecoveryDispatchContext | None,
        request: dict[str, Any],
    ) -> OrchestrationAction | None | bool:
        if applies_decision_id is None:
            return None
        authority = await db.get(OrchestrationAuthorityDecision, applies_decision_id)
        proposal_action_id = getattr(authority, "related_action_id", None) if authority is not None else None
        if proposal_action_id is None:
            return None if context is None else False
        proposal = await db.get(OrchestrationAction, proposal_action_id)
        if proposal is None or proposal.run_id != run.id or proposal.action_type != "recovery_proposal":
            return None if context is None else False
        if (
            context is None
            or context.authority_decision_id != authority.id
            or authority.status != "answered"
            or authority.selected_option != "approve_fix"
        ):
            return False
        proposal_request = proposal.request or {}
        proposal_contract = proposal.dispatch_contract or {}
        if str(proposal_request.get("proposal_id")) != str(context.proposal_id):
            return False
        proposal_hash = proposal_contract.get("proposal_hash")
        if not isinstance(proposal_hash, str) or proposal_hash != self._canonical_request_hash(proposal_request):
            return False
        try:
            recovery_proposal = RecoveryProposalV1.model_validate(proposal_request)
        except ValueError:
            return False
        if context.purpose == "apply":
            matched = [call for call in recovery_proposal.calls if call.call_id == context.call_id]
            if len(matched) != 1:
                return False
            sealed = [change.after for change in matched[0].changes if change.target_type == "baseline_repair_admission"]
            carried = context.baseline_repair_admission
            if len(sealed) > 1 or (sealed and (carried is None or carried.model_dump(mode="json") != sealed[0])) \
                    or (carried is not None and not sealed):
                return False
        elif context.approved_retry_ordinal > recovery_proposal.retry.max_attempts:
            return False
        wrapper_key = recovery_wrapper_key(context)
        wrapper = await self._service._existing_action_for_key(db, run.id, wrapper_key)
        expected_type = f"recovery_{context.purpose}"
        wrapper_request = wrapper.request if wrapper is not None else {}
        if (
            wrapper is None
            or wrapper.action_type != expected_type
            or str(wrapper_request.get("proposal_action_id")) != str(proposal.id)
            or wrapper_request.get("proposal_hash") != proposal_hash
            or (
                context.purpose == "apply"
                and wrapper_request.get("call_id") != context.call_id
            )
            or (
                context.purpose == "retry"
                and wrapper_request.get("approved_retry_ordinal") != context.approved_retry_ordinal
            )
        ):
            return False
        request_hash = self._canonical_request_hash(request)
        contract = wrapper.dispatch_contract or {}
        if contract.get("canonical_request_hash") != request_hash:
            return False
        effect_id, effect_key = contract.get("effect_action_id"), contract.get("effect_idempotency_key")
        if (effect_id is None) != (effect_key is None):
            return False
        if effect_id is not None:
            if effect_key != recovery_effect_key(context):
                return False
            effect = await self._service._existing_action_for_key(db, run.id, effect_key)
            if effect is None or str(effect.id) != str(effect_id):
                return False
        return wrapper

    @staticmethod
    def _savepoint(db: AsyncSession):
        return db.begin_nested()

    _MAX_REJECTED_ATTEMPTS = 3

    async def _attempt_key(
        self, db: AsyncSession, run_id: uuid.UUID, key: str, decision_id: uuid.UUID, fenced: bool,
    ) -> tuple[str, OrchestrationAction | None]:
        """A previously rejected (failed) action must not block the same decision forever:
        retry under a new attempt key so the executor re-checks its preconditions, up to a cap.
        Returns (key, None), or (key, latest failed action) once the cap is reached."""
        probe = key
        if fenced:
            probe, *_ = await self._service._steering_action_fence(db, run_id, key, decision_id)
        status = await db.scalar(select(OrchestrationAction.status).where(
            OrchestrationAction.run_id == run_id, OrchestrationAction.idempotency_key == probe,
        ))
        if status != "failed":
            return key, None
        failed = (await db.scalars(select(OrchestrationAction).where(
            OrchestrationAction.run_id == run_id,
            OrchestrationAction.status == "failed",
            or_(
                OrchestrationAction.idempotency_key == key,
                OrchestrationAction.idempotency_key.startswith(key + ":", autoescape=True),
            ),
        ).order_by(OrchestrationAction.created_at.desc(), OrchestrationAction.id.desc()))).all()
        progress_at = await db.scalar(select(func.max(OrchestrationAction.created_at)).where(
            OrchestrationAction.run_id == run_id, OrchestrationAction.status == "completed",
        ))
        since_progress = [a for a in failed if progress_at is None or a.created_at > progress_at]
        if len(since_progress) >= self._MAX_REJECTED_ATTEMPTS:
            return key, failed[0]
        return f"{key}:rejected:{len(failed)}", None

    async def _record_rejection(
        self, db: AsyncSession, run_id: uuid.UUID, decision_id: uuid.UUID,
        action_type: str, request: dict[str, Any], key: str, error: str,
    ) -> OrchestrationAction:
        """After the savepoint rolled back, durably keep only a failed action row."""
        action = await self._service.reserve_action(
            db, run_id=run_id, idempotency_key=key, action_type=action_type,
            request=request, decision_id=decision_id,
        )
        if action.status == "reserved":
            await self._service._fail_reserved_action_in_transaction(db, action, error)
        if error.startswith("plan_author_unavailable"):
            await self._service.planning_preflight_trigger(db, run_id, error)
        return action

    async def _corrective_trigger(
        self, db: AsyncSession, run: OrchestrationRun, decision: OrchestrationDecision,
        action_type: str, request: dict[str, Any],
    ) -> OrchestrationAction | None:
        """One open lineage per corrective request: re-selecting the same fix while its proposal is still
        pending, or the action an open lineage's sealed retry re-runs, reuses that trigger instead of starting
        another."""
        basis = {"action_type": action_type, "request": {k: v for k, v in request.items() if k != "reason"}}
        fingerprint = hashlib.sha256(json.dumps(basis, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        existing = (await self._service.open_recovery_trigger(db, run.id, fingerprint)
                    or await self._service.open_recovery_trigger_for_action(db, run.id, action_type, request))
        if existing is not None:
            return existing
        trigger, _created = await self._service.create_recovery_trigger(
            db, run, source_kind="reported_blocker", source_id=decision.id, blocker_code=f"corrective_{action_type}",
            summary=(f"The orchestrator selected {action_type} after a durable failure; "
                     "a corrective fix needs approval before it runs."),
            retry_target={"kind": "decision_loop"}, fingerprint_basis=basis,
        )
        return trigger

    @staticmethod
    async def _resolve_applied_decision(
        db: AsyncSession, run: OrchestrationRun, raw_id: Any,
    ) -> tuple[uuid.UUID | None, str | None]:
        """Return (decision_id, None) when it is an answered decision of this run,
        else (None, rejection_reason)."""
        try:
            decision_id = uuid.UUID(str(raw_id))
        except ValueError:
            return None, "applies_decision_id must be a UUID"
        authority = await db.get(OrchestrationAuthorityDecision, decision_id)
        if authority is None or authority.run_id != run.id:
            return None, "applies_decision_id does not reference a decision in this run"
        if authority.status != "answered":
            return None, f"applies_decision_id references a {authority.status} decision, not an answered one"
        return decision_id, None

    # action_type -> OrchestrationService method name. Only the actions the
    # authorized-execution loop is allowed to take on an Outcome run.
    _EXECUTORS: dict[str, str] = {
        "noop": "execute_noop_action",
        "request_plan": "execute_request_plan_action",
        "request_roadmap_replan": "execute_request_roadmap_replan_action",
        "request_plan_revision": "execute_request_plan_revision_action",
        "accept_plan": "execute_accept_plan_action",
        "create_delegation_task": "execute_create_delegation_task_action",
        "request_verification": "execute_request_verification_action",
        "retry_task": "execute_retry_task_action",
        "reassign_task": "execute_reassign_task_action",
        "schedule_meeting": "execute_schedule_meeting_action",
        "start_graph": "execute_start_graph_action",
        "ask_human": "execute_ask_human_action",
        "pause_run": "execute_pause_run_action",
        "record_warning": "execute_record_warning_action",
        "suggest_agent": "execute_suggest_agent_action",
        "recover_blocker": "execute_recover_blocker_action",
    }
