from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from decimal import Decimal

from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.base import _utcnow
from huddleroom.models.orchestration import (
    OrchestrationAction, OrchestrationBudgetReservation, OrchestrationGate, OrchestrationGoal, OrchestrationRun, OrchestrationWait,
    compute_goal_contract_fingerprint,
)
from huddleroom.models.orchestration_process import AUTHORITIES, OrchestrationAuthorityDecision
from huddleroom.models.task import Task
from huddleroom.models.user import User
from huddleroom.schemas.orchestration import ROADMAP_BUDGET_DIMENSIONS, AuthorizationResult, AuthorizationScope
from huddleroom.services.event_bus import emit_event_once
from huddleroom.services.orchestration_linkage import check_goal_linkage


RECOVERY_DECISION_KEY_PREFIX = "recovery:"
RECOVERY_GAP_DECISION_KEY_PREFIX = "recovery_gap:"
# Every authority decision that owns a reserved recovery action; cancel only via cancel_recovery_decision.
RECOVERY_OWNED_KEY_PREFIXES = (RECOVERY_DECISION_KEY_PREFIX, RECOVERY_GAP_DECISION_KEY_PREFIX)
RECOVERY_APPROVAL_OPTIONS = (("approve_fix", "Approve fix"), ("reject", "Reject"))


def runtime_decision_identity(
    run_id: uuid.UUID, subject: str, authority: str, contract_version: str
) -> str:
    raw = json.dumps([str(run_id), subject, authority, contract_version], separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


@dataclass(frozen=True)
class RuntimeAnswerResult:
    decision: OrchestrationAuthorityDecision
    continuation_applied: bool


def _option_keys(options: list | None) -> set[str]:
    """Extract the key set from an *already-validated* options list.

    Every entry is guaranteed to be a str or a dict with "key" by
    `OrchestrationAuthorityDecisionService._validate_options`, which runs at
    `create_pending` time — nothing else writes `decision.options`.
    """
    keys: set[str] = set()
    for option in options or []:
        if isinstance(option, str):
            keys.add(option)
        else:
            keys.add(str(option["key"]))
    return keys


class OrchestrationAuthorityDecisionService:
    """Authority decision lifecycle (Spec 4.3, 10.4, 10.6, 15.5).

    pending -> answered | cancelled. Creation is orchestrator-only and
    idempotent on (goal_id, decision_key) **among pending rows only**: a
    retried trigger that finds an existing pending decision for the key gets
    it back unchanged, but once a decision goes terminal (answered /
    cancelled / expired) the key is free again — a process rerun must be
    able to raise a fresh question under the same logical key (spec 6.3).
    This mirrors the partial unique index on the table; querying
    for status='pending' here keeps the service and the DB constraint in
    sync. Phase 9 wires the human answer endpoint and the manager-agent
    decision-task path onto answer_decision(). Creation shares the Phase 1
    check-then-insert caveat: single writer (orchestrator tick) until a
    concurrent creator exists.

    `authority_agent_id` pins the specific agent a non-human decision
    awaits (Spec Deviation 9): `authority` alone only names a role, so
    without this column any agent could answer in place of the intended
    one. `run_id`, `source_process_run_id`, `related_gate_id`, and
    `related_action_id` must each belong to `goal_id` (Spec Deviation 11).
    """

    @staticmethod
    def _validate_options(options: list | None) -> list:
        """Reject malformed option entries at creation time (spec 10.4:
        offered options are typed and code-validated). A str is a bare key;
        a dict must carry a non-empty "key" — anything else (wrong type, a
        dict missing "key", a blank key) is rejected here rather than
        silently dropped, which would otherwise let `answer_decision`'s
        membership check degrade to "keys is empty -> anything accepted".
        """
        if options is None:
            return []
        if not isinstance(options, list):
            raise ValueError(f"options must be a list, got {type(options).__name__}")
        if not options:
            return []
        validated: list = []
        seen: set[str] = set()
        for option in options:
            if isinstance(option, str):
                key = option
            elif isinstance(option, dict):
                key = option.get("key")
            else:
                raise ValueError(f"invalid option entry: {option!r}")
            if not key or not isinstance(key, str):
                raise ValueError(f"invalid option entry: {option!r}")
            if key in seen:
                raise ValueError(f"duplicate option key: {key!r}")
            seen.add(key)
            validated.append(option)
        return validated

    async def get_pending_by_key(
        self, db: AsyncSession, goal_id: uuid.UUID, decision_key: str
    ) -> OrchestrationAuthorityDecision | None:
        result = await db.execute(
            select(OrchestrationAuthorityDecision).where(
                OrchestrationAuthorityDecision.goal_id == goal_id,
                OrchestrationAuthorityDecision.decision_key == decision_key,
                OrchestrationAuthorityDecision.status == "pending",
            )
        )
        return result.scalar_one_or_none()

    async def create_pending(
        self,
        db: AsyncSession,
        goal_id: uuid.UUID,
        *,
        decision_key: str,
        title: str,
        question: str,
        authority: str,
        authority_agent_id: uuid.UUID | None = None,
        options: list | None = None,
        context: str | None = None,
        recommendation: str | None = None,
        consequences: str | None = None,
        run_id: uuid.UUID | None = None,
        source_process_run_id: uuid.UUID | None = None,
        related_gate_id: uuid.UUID | None = None,
        related_action_id: uuid.UUID | None = None,
    ) -> OrchestrationAuthorityDecision:
        if authority not in AUTHORITIES:
            raise ValueError(f"unknown authority: {authority!r}")
        if not decision_key:
            raise ValueError("decision_key must not be empty")
        # Spec Deviation 9: authority alone only names a role; a non-human
        # decision must always name the specific agent it awaits, and a
        # human decision must not carry one (nothing to pin it to).
        if authority == "human":
            if authority_agent_id is not None:
                raise ValueError("human-authority decisions must not set authority_agent_id")
        else:
            if authority_agent_id is None:
                raise ValueError(
                    f"{authority!r}-authority decisions require authority_agent_id"
                )
        await check_goal_linkage(
            db,
            goal_id,
            run_id=run_id,
            source_process_run_id=source_process_run_id,
            related_gate_id=related_gate_id,
            related_action_id=related_action_id,
        )
        options = self._validate_options(options)
        option_keys = _option_keys(options)
        if recommendation is not None and recommendation not in option_keys:
            raise ValueError(f"recommendation {recommendation!r} is not one of the offered options")
        existing = await self.get_pending_by_key(db, goal_id, decision_key)
        if existing is not None:
            return existing
        decision = OrchestrationAuthorityDecision(
            goal_id=goal_id,
            run_id=run_id,
            decision_key=decision_key,
            title=title,
            authority=authority,
            authority_agent_id=authority_agent_id,
            source_process_run_id=source_process_run_id,
            question=question,
            context=context,
            options=options,
            recommendation=recommendation,
            consequences=consequences,
            related_gate_id=related_gate_id,
            related_action_id=related_action_id,
        )
        try:
            async with db.begin_nested():
                db.add(decision)
                await db.flush()
        except IntegrityError:
            # Lost the check-then-insert race: a concurrent writer created
            # a pending decision for this (goal_id, decision_key) between our
            # pre-check and insert. The SAVEPOINT confined the failure; retry
            # as an idempotent fetch.
            existing = await self.get_pending_by_key(db, goal_id, decision_key)
            if existing is None:
                raise
            return existing
        return decision

    async def create_runtime_question(
        self,
        db: AsyncSession,
        goal_id: uuid.UUID,
        *,
        run_id: uuid.UUID,
        subject: str,
        authority: str,
        contract_version: str,
        continuation: dict,
        question: str,
        options: list[dict],
        context: str | None = None,
    ) -> OrchestrationAuthorityDecision:
        if authority != "human":
            raise ValueError("runtime questions require human authority")
        if not all(isinstance(value, str) and value.strip() for value in (subject, contract_version, question)):
            raise ValueError("runtime question identity fields must be non-empty")
        if not isinstance(continuation, dict):
            raise ValueError("runtime continuation must be an object")
        if set(continuation) - {"action_type", "reason"} or continuation.get("action_type") != "continue":
            raise ValueError("runtime continuation must be a continue marker")
        if "reason" in continuation and (not isinstance(continuation["reason"], str) or not continuation["reason"].strip()):
            raise ValueError("runtime continuation reason must be a non-empty string")
        options = self._validate_options(options)
        if not options:
            raise ValueError("runtime questions require offered options")
        await check_goal_linkage(db, goal_id, run_id=run_id)
        identity = runtime_decision_identity(run_id, subject, authority, contract_version)
        existing = await db.scalar(select(OrchestrationAuthorityDecision).where(
            OrchestrationAuthorityDecision.runtime_identity == identity
        ))
        if existing is not None:
            return existing
        decision = OrchestrationAuthorityDecision(
            goal_id=goal_id,
            run_id=run_id,
            decision_key=f"runtime:{identity}",
            title=question[:255],
            authority=authority,
            runtime_identity=identity,
            contract_version=contract_version,
            continuation=continuation,
            question=question,
            context=context,
            options=options,
        )
        try:
            async with db.begin_nested():
                db.add(decision)
                await db.flush()
        except IntegrityError:
            existing = await db.scalar(select(OrchestrationAuthorityDecision).where(
                OrchestrationAuthorityDecision.runtime_identity == identity
            ))
            if existing is None:
                raise
            return existing
        return decision

    async def answer_runtime_question(
        self,
        db: AsyncSession,
        decision: OrchestrationAuthorityDecision,
        selected_option: str,
        *,
        actor_user_id: uuid.UUID | None,
        contract_version: str,
        reason: str | None = None,
    ) -> RuntimeAnswerResult:
        """Answer one versioned runtime question and reserve its marker once."""
        if actor_user_id is None or decision.authority != "human":
            raise ValueError("runtime questions require an authenticated human actor")
        if await db.get(User, actor_user_id) is None:
            raise ValueError("runtime questions require an authenticated human actor")
        if not selected_option or selected_option not in _option_keys(decision.options):
            raise ValueError("selected_option is not among the offered options")
        # A stale request is deliberately inert, including for already answered rows.
        if decision.contract_version != contract_version:
            return RuntimeAnswerResult(decision=decision, continuation_applied=False)

        from huddleroom.services.orchestration_service import OrchestrationService

        orchestration = OrchestrationService()
        async with orchestration._lock_goal_for_baseline_transition(db, decision.goal_id):
            current = await db.get(OrchestrationAuthorityDecision, decision.id, populate_existing=True)
            if current is None:
                raise ValueError("runtime decision no longer exists")
            if current.contract_version != contract_version:
                return RuntimeAnswerResult(decision=current, continuation_applied=False)
            if current.runtime_identity != current.decision_key.removeprefix("runtime:"):
                raise ValueError("runtime decision identity is invalid")
            normalized_reason = reason.strip() if reason is not None else None
            if normalized_reason is not None and len(normalized_reason) > 4000:
                raise ValueError("runtime answer reason must not exceed 4000 codepoints")
            if len(current.options or []) == 1 and _option_keys(current.options) == {"acknowledge"}:
                if not normalized_reason:
                    raise ValueError("runtime text answers require a non-blank reason")
            run = await db.get(OrchestrationRun, current.run_id, populate_existing=True)
            if current.status == "answered":
                if current.selected_option != selected_option or current.reason != normalized_reason:
                    raise ValueError("conflicting duplicate runtime answer")
            elif current.status != "pending":
                raise ValueError(f"cannot answer decision in status {current.status!r}")
            else:
                result = await db.execute(update(OrchestrationAuthorityDecision).where(
                    OrchestrationAuthorityDecision.id == current.id,
                    OrchestrationAuthorityDecision.status == "pending",
                ).values(
                    status="answered", selected_option=selected_option, reason=normalized_reason,
                    decided_by_user_id=actor_user_id, decided_at=_utcnow(),
                ))
                if result.rowcount != 1:
                    await db.refresh(current)
                    if (current.status == "answered" and current.selected_option == selected_option
                            and current.reason == normalized_reason):
                        return RuntimeAnswerResult(decision=current, continuation_applied=False)
                    raise ValueError("conflicting duplicate runtime answer")
                await db.refresh(current)

            goal = await db.get(OrchestrationGoal, current.goal_id, populate_existing=True)
            await emit_event_once(
                db, goal.project_id, "authority.decision_resolved",
                {"decision_id": str(current.id), "status": current.status}, source="orchestrator",
                dedup_key=f"authority.decision_resolved:{current.id}",
            )

            run = await db.get(OrchestrationRun, current.run_id, populate_existing=True)
            if run is None or goal is None or goal.status in {"paused", "cancelled"} or run.status not in {"running", "blocked"} or (
                goal.goal_type == "continuous" and (goal.continuous_state or {}).get("stopped_at")
            ):
                return RuntimeAnswerResult(decision=current, continuation_applied=False)

            decision_id = str(current.id)
            run.active_blockers = [
                blocker for blocker in (run.active_blockers or [])
                if not isinstance(blocker, dict) or blocker.get("decision_id") != decision_id
            ]
            if not run.active_blockers:
                if goal.status == "blocked":
                    goal.status = "active"
                if run.status == "blocked":
                    run.status = "running"
            waits = await db.scalars(select(OrchestrationWait).where(
                OrchestrationWait.run_id == run.id, OrchestrationWait.status == "open"
            ))
            for wait in waits:
                if decision_id in {
                    str((wait.owner or {}).get("decision_id", "")),
                    str((wait.awaited_event or {}).get("decision_id", "")),
                }:
                    wait.status, wait.cleared_at = "cleared", _utcnow()
            key = f"run:{run.id}:kind:decision_continuation:decision:{current.id}"
            action = await orchestration.reserve_action(
                db, run.id, key, "decision_continuation", current.continuation or {}
            )
            if action.status == "reserved":
                action = await orchestration._mark_action_completed(
                    db, action, target_type="authority_decision", target_id=current.id
                )
            if action.status != "completed":
                return RuntimeAnswerResult(decision=current, continuation_applied=False)
            linked = await db.execute(update(OrchestrationAuthorityDecision).where(
                OrchestrationAuthorityDecision.id == current.id,
                OrchestrationAuthorityDecision.continuation_action_id.is_(None),
            ).values(continuation_action_id=action.id, continuation_applied_at=_utcnow()))
            await db.refresh(current)
            return RuntimeAnswerResult(decision=current, continuation_applied=linked.rowcount == 1)

    async def create_recovery_approval(
        self,
        db: AsyncSession,
        goal_id: uuid.UUID,
        *,
        run_id: uuid.UUID,
        proposal_action_id: uuid.UUID,
        proposal_id: uuid.UUID,
        proposal_hash: str,
        question: str,
        context: str,
    ) -> OrchestrationAuthorityDecision:
        """One human approve_fix/reject decision bound to the exact proposal hash."""
        return await self.create_pending(
            db, goal_id,
            decision_key=f"{RECOVERY_DECISION_KEY_PREFIX}{proposal_id}:{proposal_hash}",
            title="Fix proposal", question=question, authority="human",
            options=[{"key": key, "label": label} for key, label in RECOVERY_APPROVAL_OPTIONS],
            context=context, run_id=run_id, related_action_id=proposal_action_id,
        )

    async def answer_recovery_decision(
        self,
        db: AsyncSession,
        decision: OrchestrationAuthorityDecision,
        *,
        selected_option: str,
        reason: str | None,
        decided_by_user_id: uuid.UUID,
    ) -> OrchestrationAuthorityDecision:
        """Record a recovery approval answer; callers own the drift check and goal lock."""
        if not decision.decision_key.startswith(RECOVERY_DECISION_KEY_PREFIX):
            raise ValueError("not a recovery proposal decision")
        if selected_option not in {key for key, _ in RECOVERY_APPROVAL_OPTIONS}:
            raise ValueError("recovery proposals accept only approve_fix or reject")
        if selected_option == "reject" and not (reason and reason.strip()):
            raise ValueError("rejecting a fix proposal requires a reason")
        return await self._answer(
            db, decision, selected_option=selected_option, reason=reason,
            decided_by_user_id=decided_by_user_id, decided_by_agent_id=None,
        )

    async def answer_decision(
        self,
        db: AsyncSession,
        decision: OrchestrationAuthorityDecision,
        *,
        selected_option: str,
        reason: str | None = None,
        decided_by_user_id: uuid.UUID | None = None,
        decided_by_agent_id: uuid.UUID | None = None,
        control_writer: bool = False,
    ) -> OrchestrationAuthorityDecision:
        # Recovery approvals must pass the proposal drift check first (spec: answer handling
        # recognizes the recovery key prefix); the generic path can never approve one.
        if decision.decision_key.startswith(RECOVERY_DECISION_KEY_PREFIX):
            raise ValueError("recovery proposals must be answered through blocker recovery")
        return await self._answer(
            db, decision, selected_option=selected_option, reason=reason,
            decided_by_user_id=decided_by_user_id, decided_by_agent_id=decided_by_agent_id,
            control_writer=control_writer,
        )

    async def _answer(
        self,
        db: AsyncSession,
        decision: OrchestrationAuthorityDecision,
        *,
        selected_option: str,
        reason: str | None,
        decided_by_user_id: uuid.UUID | None,
        decided_by_agent_id: uuid.UUID | None,
        control_writer: bool = False,
    ) -> OrchestrationAuthorityDecision:
        if decision.status != "pending":
            raise ValueError(f"cannot answer decision in status {decision.status!r}")
        # Enforce the decision's declared authority (spec 4.3, 6.5, 10.4):
        # "human" must be answered by a user; every other authority
        # ("manager", "team_lead", "agent") must be answered by an agent.
        if decision.authority == "human":
            if decided_by_user_id is None or decided_by_agent_id is not None:
                raise ValueError("human-authority decisions require decided_by_user_id only")
        else:
            if decided_by_agent_id is None or decided_by_user_id is not None:
                raise ValueError(
                    f"{decision.authority!r}-authority decisions require decided_by_agent_id only"
                )
            # authority alone only names a role; the decision names the
            # specific agent instance it awaits (Spec Deviation 9). A
            # different agent — even one with the right role — may not
            # answer in its place.
            if decided_by_agent_id != decision.authority_agent_id:
                raise ValueError(
                    "decided_by_agent_id does not match the decision's authority_agent_id"
                )
        if not selected_option or not isinstance(selected_option, str) or not selected_option.strip():
            raise ValueError("selected_option must be a non-empty string")
        keys = _option_keys(decision.options)
        if keys and selected_option not in keys:
            raise ValueError(f"selected_option {selected_option!r} is not among the offered options")
        # Lazy imports avoid the cycle from orchestration_service back to this service.
        from huddleroom.services.orchestration_service import OrchestrationService
        from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def answer_boundary():
            if control_writer:
                yield
            else:
                async with OrchestrationService()._lock_goal_for_baseline_transition(db, decision.goal_id):
                    yield

        async with answer_boundary():
            # Status-conditional UPDATE (not a load-then-write) so two concurrent
            # answer/cancel calls on the same pending decision can't both
            # succeed: only the first to land wins the row, the second sees
            # rowcount 0 and fails instead of silently overwriting terminal data.
            overrides_recommendation = (
                decision.recommendation is not None
                and selected_option != decision.recommendation
            )
            result = await db.execute(
                update(OrchestrationAuthorityDecision)
                .where(
                    OrchestrationAuthorityDecision.id == decision.id,
                    OrchestrationAuthorityDecision.status == "pending",
                )
                .values(
                    status="answered",
                    selected_option=selected_option,
                    reason=reason,
                    decided_by_user_id=decided_by_user_id,
                    decided_by_agent_id=decided_by_agent_id,
                    overrides_recommendation=overrides_recommendation,
                    decided_at=_utcnow(),
                )
            )
            if result.rowcount != 1:
                raise ValueError("cannot answer decision: it was concurrently modified")
            await db.refresh(decision)
            if overrides_recommendation and decision.source_process_run_id is None:
                warning = await OrchestrationWarningService().create_warning(
                    db,
                    decision.goal_id,
                    warning_type="authority_decision_overrode_recommendation",
                    severity="warning",
                    message=(
                        f"Decision '{decision.title}' was answered with {selected_option!r} "
                        f"instead of the recommended {decision.recommendation!r}: "
                        f"{reason or 'no reason given'}"
                    ),
                    run_id=decision.run_id,
                    source_process_run_id=decision.source_process_run_id,
                    related_gate_id=decision.related_gate_id,
                    related_action_id=decision.related_action_id,
                    related_authority_decision_id=decision.id,
                )
                await db.execute(
                    update(OrchestrationAuthorityDecision)
                    .where(OrchestrationAuthorityDecision.id == decision.id)
                    .values(created_warning_id=warning.id)
                )
                await db.refresh(decision)
            goal = await db.get(OrchestrationGoal, decision.goal_id)
            await emit_event_once(
                db, goal.project_id, "authority.decision_resolved",
                {"decision_id": str(decision.id), "status": decision.status}, source="orchestrator",
                dedup_key=f"authority.decision_resolved:{decision.id}",
            )
            return decision

    async def cancel_decision(
        self,
        db: AsyncSession,
        decision: OrchestrationAuthorityDecision,
        *,
        reason: str,
    ) -> OrchestrationAuthorityDecision:
        # A bare cancel would leave the linked recovery proposal / capability gap reserved forever.
        if decision.decision_key.startswith(RECOVERY_OWNED_KEY_PREFIXES):
            raise ValueError("recovery proposals must be cancelled through cancel_recovery_decision")
        return await self._cancel(db, decision, reason=reason)

    async def cancel_recovery_decision(
        self,
        db: AsyncSession,
        decision: OrchestrationAuthorityDecision,
        *,
        reason: str,
    ) -> OrchestrationAuthorityDecision:
        """Cancel a recovery approval or capability-gap decision and fail its reserved proposal/gap action
        with `reason` as the error."""
        if not decision.decision_key.startswith(RECOVERY_OWNED_KEY_PREFIXES):
            raise ValueError("not a recovery proposal decision")
        decision = await self._cancel(db, decision, reason=reason)
        if decision.related_action_id is not None:
            from huddleroom.models.orchestration import OrchestrationAction
            from huddleroom.services.orchestration_service import OrchestrationService

            proposal = await db.get(OrchestrationAction, decision.related_action_id)
            if proposal is not None and proposal.status == "reserved":
                await OrchestrationService().mark_action_failed(db, proposal, reason)
        return decision

    async def _cancel(
        self,
        db: AsyncSession,
        decision: OrchestrationAuthorityDecision,
        *,
        reason: str,
    ) -> OrchestrationAuthorityDecision:
        if decision.status != "pending":
            raise ValueError(f"cannot cancel decision in status {decision.status!r}")
        if not reason or not reason.strip():
            raise ValueError("reason must not be empty")
        result = await db.execute(
            update(OrchestrationAuthorityDecision)
            .where(
                OrchestrationAuthorityDecision.id == decision.id,
                OrchestrationAuthorityDecision.status == "pending",
            )
            .values(status="cancelled", reason=reason)
        )
        if result.rowcount != 1:
            raise ValueError("cannot cancel decision: it was concurrently modified")
        await db.refresh(decision)
        run = await db.get(OrchestrationRun, decision.run_id) if decision.run_id else None
        if run is not None and run.active_blockers:  # a cancelled owner ask no longer blocks its subject
            run.active_blockers = [
                blocker for blocker in run.active_blockers
                if not (isinstance(blocker, dict) and blocker.get("kind") == "owner_escalation"
                        and blocker.get("decision_id") == str(decision.id))
            ]
        goal = await db.get(OrchestrationGoal, decision.goal_id)
        await emit_event_once(
            db, goal.project_id, "authority.decision_resolved",
            {"decision_id": str(decision.id), "status": decision.status}, source="orchestrator",
            dedup_key=f"authority.decision_resolved:{decision.id}",
        )
        return decision

    async def link_delegation_action(
        self,
        db: AsyncSession,
        decision: OrchestrationAuthorityDecision,
        *,
        action_id: uuid.UUID,
    ) -> OrchestrationAuthorityDecision:
        """Record the delegation-task action created to deliver a non-human
        pending decision to its named agent (spec 8.3.1). Idempotent: calling
        this again with the same action_id is a no-op; calling it with a
        different one while already linked is rejected rather than silently
        overwriting which task the agent is expected to report against.
        """
        if decision.related_action_id == action_id:
            return decision
        if decision.related_action_id is not None:
            raise ValueError(
                f"decision {decision.id} is already linked to action {decision.related_action_id}"
            )
        result = await db.execute(
            update(OrchestrationAuthorityDecision)
            .where(
                OrchestrationAuthorityDecision.id == decision.id,
                OrchestrationAuthorityDecision.related_action_id.is_(None),
            )
            .values(related_action_id=action_id)
        )
        if result.rowcount != 1:
            await db.refresh(decision)
            if decision.related_action_id == action_id:
                return decision
            raise ValueError(f"cannot link delegation action: decision {decision.id} was concurrently modified")
        await db.refresh(decision)
        return decision

    async def list_decisions(
        self,
        db: AsyncSession,
        goal_id: uuid.UUID,
        *,
        status: str | None = None,
    ) -> list[OrchestrationAuthorityDecision]:
        query = select(OrchestrationAuthorityDecision).where(
            OrchestrationAuthorityDecision.goal_id == goal_id
        )
        if status is not None:
            query = query.where(OrchestrationAuthorityDecision.status == status)
        query = query.order_by(
            OrchestrationAuthorityDecision.asked_at.asc(),
            OrchestrationAuthorityDecision.created_at.asc(),
            OrchestrationAuthorityDecision.id.asc(),
        )
        result = await db.execute(query)
        return list(result.scalars().all())


_WORKSPACE_RANK = {"read_only": 0, "staged": 1, "direct": 2}
_GRANT_SCOPES = frozenset({"planning_only", "execution"})


@dataclass(frozen=True)
class ExecutionGrant:
    """A completed, server-owned execution grant for one run."""

    action: object
    scope: str
    legacy: bool = False


def _run_scope_version(run: OrchestrationRun) -> int | None:
    state = run.plan_state if isinstance(run.plan_state, dict) else {}
    return state.get("scope_contract_version") if isinstance(state.get("scope_contract_version"), int) else None


async def execution_grant(
    db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun,
) -> ExecutionGrant | None:
    """Return only a valid explicit execution grant (or a genuinely old one).

    New runs carry a marker in their existing JSON state.  That lets historical
    unscoped actions retain their meaning without allowing newly-created child
    actions to masquerade as history.
    """
    from huddleroom.services.orchestration_service import OrchestrationService

    service = OrchestrationService()
    action = await service._existing_action_for_key(
        db, run.id, f"run:{run.id}:kind:authorize_execution",
    )
    if action is None or action.status != "completed":
        return None
    request = action.request if isinstance(action.request, dict) else {}
    scope = request.get("scope")
    marked = _run_scope_version(run) == 1
    if not marked:
        # Only the pre-G3 representation has compatibility semantics.
        return ExecutionGrant(action, "execution", legacy=True) if scope is None else (
            ExecutionGrant(action, scope) if scope in _GRANT_SCOPES else None
        )
    if scope not in _GRANT_SCOPES or request.get("contract_fingerprint") != run.goal_contract_fingerprint:
        return None
    if scope == "planning_only":
        base_action = action
        upgrade_action = await service._existing_action_for_key(
            db, run.id, execution_upgrade_key(run.id, run.goal_contract_fingerprint),
        )
        upgrade = upgrade_action.request if upgrade_action is not None and isinstance(upgrade_action.request, dict) else {}
        if upgrade_action is None or upgrade_action.status != "completed" or upgrade.get("scope") != "execution":
            return ExecutionGrant(base_action, "planning_only")
        action = upgrade_action
        scope = "execution"
    if goal.parent_goal_id is not None:
        source_run_id = request.get("source_run_id")
        source_goal_id = request.get("source_goal_id")
        source_action_id = request.get("source_action_id")
        if not all(isinstance(value, str) for value in (source_run_id, source_goal_id, source_action_id)):
            return None
        try:
            source_run = await db.get(OrchestrationRun, uuid.UUID(source_run_id))
            source_goal = await db.get(OrchestrationGoal, uuid.UUID(source_goal_id))
            source_action = await db.get(type(action), uuid.UUID(source_action_id))
        except ValueError:
            return None
        if (source_run is None or source_goal is None or source_action is None
                or source_run.goal_id != source_goal.id or source_action.run_id != source_run.id):
            return None
        source_grant = await execution_grant(db, source_goal, source_run)
        if source_grant is None or source_grant.scope != "execution" or source_grant.action.id != source_action.id:
            return None
    return ExecutionGrant(action, scope)


async def inherit_execution_grant(
    db: AsyncSession,
    source_goal: OrchestrationGoal,
    source_run: OrchestrationRun,
    target_goal: OrchestrationGoal,
    target_run: OrchestrationRun,
    *,
    actor: str,
) -> ExecutionGrant | None:
    """Create a lineage-bound child grant only from current execution authority."""
    from huddleroom.models.orchestration import OrchestrationAction
    from huddleroom.services.orchestration_service import OrchestrationService

    source = await execution_grant(db, source_goal, source_run)
    if source is None or source.scope != "execution":
        return None
    fingerprint = compute_goal_contract_fingerprint(target_goal)
    if target_run.goal_contract_fingerprint not in (None, fingerprint):
        return None
    target_run.goal_contract_fingerprint = fingerprint
    state = target_run.plan_state if isinstance(target_run.plan_state, dict) else {}
    target_run.plan_state = {**state, "scope_contract_version": 1}
    service = OrchestrationService()
    request = {
        "action_type": "authorize_execution", "actor": actor, "scope": "execution",
        "contract_fingerprint": fingerprint, "source_goal_id": str(source_goal.id),
        "source_run_id": str(source_run.id), "source_action_id": str(source.action.id),
    }
    key = service._authorize_execution_key(target_run.id)
    action = await service._existing_action_for_key(db, target_run.id, key)
    if action is None and target_goal.parent_goal_id is not None:
        action = OrchestrationAction(
            id=uuid.uuid5(uuid.NAMESPACE_URL, f"rally:authorize-child:{target_run.id}"),
            run_id=target_run.id, idempotency_key=key, action_type="authorize_execution",
            status="reserved", request=request,
        )
        db.add(action)
        await db.flush()
    elif action is None:
        action = await service.reserve_action(db, target_run.id, key, "authorize_execution", request)
    elif action.request != request:
        return None
    if action.status == "reserved":
        await service._mark_action_completed(db, action, target_type="run", target_id=target_run.id)
    return await execution_grant(db, target_goal, target_run)


def reauthorize_execution_key(run_id: uuid.UUID, fingerprint: str) -> str:
    """Re-Start after a contract change; the run's frozen fingerprint is never rewritten."""
    return f"run:{run_id}:kind:reauthorize_execution:{fingerprint}"


def execution_upgrade_key(run_id: uuid.UUID, fingerprint: str) -> str:
    return f"run:{run_id}:kind:grant_execution:{fingerprint}"


async def effective_authorization(
    db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun, task: Task | None = None,
) -> AuthorizationResult:
    """Read-only effective authority for a run (and task); the first failing check denies.

    Start is orchestration/model-spend authority only. No goal-level external-write or
    live-send grants exist, so those stay empty; generic shell effects are not policed here.
    Capacity is not an input (capacity_wait is owned by the scheduler).
    """
    # Lazy imports avoid the cycle from orchestration_service back to this service.
    from huddleroom.services.orchestration_budget_service import (
        BudgetMeasurementError, OrchestrationBudgetService, canonical_amounts, parent_caps,
    )
    from huddleroom.services.orchestration_service import OrchestrationService
    from huddleroom.services.session_service import SessionService

    svc = OrchestrationService()
    obj = svc._json_object_or_empty
    fp = compute_goal_contract_fingerprint(goal)
    if goal.parent_goal_id is not None:
        source = "parent_release"
    elif run.phase == "baseline":  # ready is baseline-complete: only Start authorizes execution
        source = "baseline"
    else:
        source = "root_start"
    context = await SessionService()._immutable_roadmap_context(db, task) if task is not None else obj(
        goal.orchestrator_context
    )
    roadmap = obj(context.get("roadmap"))
    if source == "baseline":
        workspace = "read_only"
    elif roadmap.get("mutates_shared_state") and roadmap.get("staging_boundary"):
        workspace = "staged"
    else:
        workspace = "direct"
    raw_scope = obj(obj(task.metadata_ if task is not None else None).get("orchestration_contract")).get(
        "authorization_scope"
    )
    scope = None
    if raw_scope is not None:
        try:
            scope = AuthorizationScope.model_validate(raw_scope)
        except ValidationError:
            scope = AuthorizationScope()  # fail closed: an unreadable declaration grants nothing
    constraints = {"no_external_writes", "no_live_sends", *(scope.tool_constraints if scope else ())}
    if roadmap.get("no_publish_before_integration"):
        constraints.add("no_publish_before_integration")
    remaining: dict[str, Decimal] = {}

    def result(reason: str | None = None, decision: str | None = None) -> AuthorizationResult:
        return AuthorizationResult(
            allowed=reason is None, reason_code=reason, required_decision=decision,
            authority_source=source, run_id=run.id, contract_fingerprint=fp,
            model_spend_remaining=remaining, workspace_mutation=workspace,
            tool_constraints=sorted(constraints),
        )

    async def completed(key: str) -> bool:
        action = await svc._existing_action_for_key(db, run.id, key)
        return action is not None and action.status == "completed"

    async def planning_control_task() -> bool:
        if task is None:
            return False
        candidate = task
        parent_id = obj(task.metadata_).get("result_correction_parent_task_id")
        if parent_id:
            try:
                candidate = await db.get(Task, uuid.UUID(str(parent_id)))
            except ValueError:
                return False
            if candidate is None:
                return False
            correction_action = await db.scalar(select(OrchestrationAction).where(
                OrchestrationAction.run_id == run.id,
                OrchestrationAction.target_type == "task",
                OrchestrationAction.target_id == task.id,
                OrchestrationAction.action_type.in_(("create_delegation_task", "result_correction")),
                OrchestrationAction.status == "completed",
            ))
            if (correction_action is None
                    or obj(correction_action.request).get("parent_task_id") != str(candidate.id)):
                return False
        orchestration = obj((candidate.metadata_ or {}).get("orchestration"))
        try:
            action_id = uuid.UUID(str(orchestration.get("action_id")))
        except (TypeError, ValueError):
            return False
        action = await db.get(OrchestrationAction, action_id)
        return bool(
            action and action.run_id == run.id and action.status == "completed" and action.target_type == "task"
            and action.target_id == candidate.id
            and action.action_type in {"request_plan", "request_plan_revision"}
        )

    async def current_candidate_verifier() -> bool:
        """The only planning-only verifier is the live, independent plan gate task."""
        if task is None:
            return False
        candidate = task
        correction_parent_id = obj(task.metadata_).get("result_correction_parent_task_id")
        if correction_parent_id:
            try:
                candidate = await db.get(Task, uuid.UUID(str(correction_parent_id)))
                correction_action_id = uuid.UUID(str(obj(obj(task.metadata_).get("orchestration")).get("action_id")))
            except (TypeError, ValueError):
                return False
            correction_action = await db.get(OrchestrationAction, correction_action_id)
            if (
                candidate is None
                or obj(candidate.metadata_).get("result_correction_parent_task_id")
                or task.assigned_to != candidate.assigned_to
                or correction_action is None
                or correction_action.run_id != run.id
                or correction_action.status != "completed"
                or correction_action.action_type != "create_delegation_task"
                or correction_action.target_type != "task"
                or correction_action.target_id != task.id
                or obj(correction_action.request).get("parent_task_id") != str(candidate.id)
                or obj(correction_action.request).get("agent_id") != str(candidate.assigned_to)
            ):
                return False
        orchestration = obj(obj(candidate.metadata_).get("orchestration"))
        action_id = orchestration.get("verification_action_id")
        gate_id = orchestration.get("verification_gate_id")
        try:
            action = await db.get(OrchestrationAction, uuid.UUID(str(action_id)))
            gate = await db.get(OrchestrationGate, uuid.UUID(str(gate_id)))
        except (TypeError, ValueError):
            return False
        if (
            action is None or gate is None or action.run_id != run.id or action.status != "completed"
            or action.action_type != "request_verification" or action.target_id != candidate.id
            or gate.run_id != run.id or gate.gate_type != "roadmap_plan_approval" or gate.status != "open"
            or obj(action.request).get("gate_id") != str(gate.id)
            or obj(action.request).get("verifier_agent_id") != str(candidate.assigned_to)
        ):
            return False
        required = obj(gate.required_evidence)
        if any(obj(action.request).get(key) != required.get(key) for key in (
            "plan_review_identity", "plan_review_identity_hash",
        )):
            return False
        from huddleroom.models.artifact import Artifact
        from huddleroom.services.orchestration_roadmap_service import OrchestrationRoadmapService
        from huddleroom.services.orchestration_roster_mapper import OrchestrationRosterMapper

        artifact_id = required.get("plan_artifact_id")
        try:
            artifact = await db.get(Artifact, uuid.UUID(str(artifact_id)))
        except (TypeError, ValueError):
            return False
        if artifact is None or candidate.assigned_to == artifact.created_by_agent:
            return False
        if candidate.assigned_to not in {agent.id for agent in await OrchestrationRosterMapper().current_active_agents(db)}:
            return False
        try:
            await OrchestrationRoadmapService(svc)._require_current_plan_review(
                db, goal, run, artifact,
                {key: required.get(key) for key in ("plan_review_identity", "plan_review_identity_hash")},
                boundary="candidate verifier authorization",
            )
        except HTTPException:
            return False
        return True

    async def current_candidate_manager() -> bool:
        if task is None or task.assigned_to != goal.manager_agent_id:
            return False
        decision_id = obj(obj(task.metadata_).get("orchestration")).get("authority_decision_id")
        try:
            decision = await db.get(OrchestrationAuthorityDecision, uuid.UUID(str(decision_id)))
            context = json.loads(decision.context or "null") if decision is not None else None
            artifact_id = uuid.UUID(str(context["plan_review_identity"]["artifact_id"]))
        except (KeyError, TypeError, ValueError):
            return False
        if (
            decision is None or decision.run_id != run.id or decision.status != "pending"
            or decision.authority_agent_id != goal.manager_agent_id
            or not decision.decision_key.startswith("roadmap_plan:")
        ):
            return False
        related = await db.get(OrchestrationAction, decision.related_action_id) if decision.related_action_id else None
        if (related is None or related.run_id != run.id or related.status != "completed"
                or related.action_type != "create_delegation_task" or related.target_id != task.id):
            return False
        orchestration = obj(obj(task.metadata_).get("orchestration"))
        if (orchestration.get("authority_decision_id") != str(decision.id)
                or obj(related.request).get("authority_decision_id") != str(decision.id)
                or orchestration.get("plan_review_identity_hash") != context.get("plan_review_identity_hash")
                or obj(related.request).get("plan_review_identity_hash") != context.get("plan_review_identity_hash")):
            return False
        from huddleroom.services.orchestration_roster_mapper import OrchestrationRosterMapper
        if task.assigned_to not in {agent.id for agent in await OrchestrationRosterMapper().current_active_agents(db)}:
            return False
        from huddleroom.models.artifact import Artifact
        from huddleroom.services.orchestration_roadmap_service import OrchestrationRoadmapService
        artifact = await db.get(Artifact, artifact_id)
        if artifact is None:
            return False
        try:
            await OrchestrationRoadmapService(svc)._require_current_plan_review(
                db, goal, run, artifact,
                {key: context.get(key) for key in ("plan_review_identity", "plan_review_identity_hash")},
                boundary="candidate manager authorization",
            )
        except HTTPException:
            return False
        return True

    if goal.status not in ("active", "blocked") or run.status not in ("running", "blocked"):
        return result("run_not_active", "resume" if "paused" in (goal.status, run.status) else None)
    budget = OrchestrationBudgetService()
    try:
        if goal.goal_type in ("roadmap", "continuous"):  # the goal is its own budget parent
            raw = (await budget.remaining(db, goal))["remaining"]
        else:
            state_caps = obj(run.budget_state).get("caps")
            caps = canonical_amounts(state_caps, allowed=ROADMAP_BUDGET_DIMENSIONS) if state_caps else parent_caps(goal)
            raw = await budget.run_remaining(db, run, caps) if caps else {}
    except BudgetMeasurementError:
        raw = {}  # ponytail: parity with the Outcome stop: measurement error is an attention blocker, not a denial
    remaining = {key: Decimal(value) for key, value in raw.items()}
    grant = await execution_grant(db, goal, run)
    if source == "parent_release":
        delta = obj(obj(run.plan_state).get("child_delta_baseline"))
        if delta and delta.get("status") != "accepted":
            return result("parent_release_pending", "parent_release")
        if grant is None:
            return result("parent_release_missing", "parent_release")
        if grant.scope != "execution":
            return result("execution_scope_required", "upgrade_execution")
        reservation = await db.scalar(select(OrchestrationBudgetReservation.status).where(
            OrchestrationBudgetReservation.parent_goal_id == goal.parent_goal_id,
            OrchestrationBudgetReservation.child_goal_id == goal.id,
        ))
        if reservation != "active":
            return result("parent_reservation_inactive", "parent_release")
        if run.goal_contract_fingerprint not in (None, fp):
            return result("contract_changed_since_release", "parent_release")
    elif source == "baseline":
        if not run.baseline_authorized:
            return result("baseline_not_authorized", "authorize_baseline")
    else:
        # A predecessor run's Start is stale by construction: it is cancelled (check above)
        # and the successor carries no authorize_execution action of its own.
        if grant is None:
            return result("start_required", "start")
        if grant.scope != "execution" and not (
            await planning_control_task() or await current_candidate_verifier() or await current_candidate_manager()
        ):
            return result("execution_scope_required", "upgrade_execution")
        # NULL fingerprint = legacy run started before freezing; grandfathered.
        if run.goal_contract_fingerprint not in (None, fp) and not await completed(
            reauthorize_execution_key(run.id, fp)
        ):
            return result("contract_changed_since_start", "start")

    if svc._budget_is_exhausted(run):  # per-claim headroom is enforced in _orchestration_budget_limits
        return result("budget_exhausted", "increase_budget")

    if task is not None and roadmap and task.assigned_to is not None:
        allowed = svc._roadmap_team_agent_ids(context.get("team"))
        agent_id = str(task.assigned_to)
        if (allowed is not None and agent_id not in allowed) or context.get("assigned_agent_id") not in (None, agent_id):
            return result("agent_outside_accepted_team", "reassign_task")

    if scope is not None:
        if not scope.model_spend.allowed or any(
            value <= 0 for value in scope.model_spend.remaining_by_dimension.values()
        ):
            return result("model_spend_not_granted", "revise_task_scope")
        for key, value in scope.model_spend.remaining_by_dimension.items():
            remaining[key] = min(remaining[key], value) if key in remaining else value
        if scope.external_writes.allowed:
            return result("external_write_not_granted", "ask_human")
        if scope.live_sends.allowed:
            return result("live_send_not_granted", "ask_human")
        if _WORKSPACE_RANK[scope.workspace_mutation] > _WORKSPACE_RANK[workspace]:
            return result("workspace_mutation_exceeds_policy", "ask_human")
        workspace = scope.workspace_mutation  # a declared scope can only narrow the goal level
    return result()
