from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.base import _utcnow
from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun, OrchestrationWait
from huddleroom.models.orchestration_process import AUTHORITIES, OrchestrationAuthorityDecision
from huddleroom.models.user import User
from huddleroom.services.event_bus import emit_event_once
from huddleroom.services.orchestration_linkage import check_goal_linkage


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
    This mirrors the partial unique index on the table (Task 2); querying
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
            title=f"Runtime decision: {subject}",
            authority=authority,
            runtime_identity=identity,
            contract_version=contract_version,
            continuation=continuation,
            question=question,
            context=f"Blocked action: {subject}",
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
            run = await db.get(OrchestrationRun, current.run_id, populate_existing=True)
            if current.status == "answered":
                if current.selected_option != selected_option:
                    raise ValueError("conflicting duplicate runtime answer")
            elif current.status != "pending":
                raise ValueError(f"cannot answer decision in status {current.status!r}")
            else:
                result = await db.execute(update(OrchestrationAuthorityDecision).where(
                    OrchestrationAuthorityDecision.id == current.id,
                    OrchestrationAuthorityDecision.status == "pending",
                ).values(
                    status="answered", selected_option=selected_option,
                    decided_by_user_id=actor_user_id, decided_at=_utcnow(),
                ))
                if result.rowcount != 1:
                    await db.refresh(current)
                    if current.status == "answered" and current.selected_option == selected_option:
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

    async def answer_decision(
        self,
        db: AsyncSession,
        decision: OrchestrationAuthorityDecision,
        *,
        selected_option: str,
        reason: str | None = None,
        decided_by_user_id: uuid.UUID | None = None,
        decided_by_agent_id: uuid.UUID | None = None,
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

        async with OrchestrationService()._lock_goal_for_baseline_transition(db, decision.goal_id):
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
