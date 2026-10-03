from __future__ import annotations

import uuid

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.base import _utcnow
from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun
from huddleroom.models.orchestration_process import WARNING_SEVERITIES, OrchestrationWarning
from huddleroom.services.orchestration_linkage import check_goal_linkage

# resolved_by value marking auto-resolution (spec 11.8); anything else is a
# manual resolution attributed "human:<user_id>" / "manager:<agent_id>".
AUTO_RESOLVED_BY = "system"

# Durable marker: the exact resolved_reason a "<process_type>_stale_inputs"
# suggestion (see suggest_stale_inputs below) must carry when a human
# dismisses it (as opposed to approving a rerun, which resolves it with a
# different reason and supersedes the process run entirely). OrchestrationWarning
# has no dedicated boolean/state column for this -- resolved_reason is the
# field the frontend's Dismiss action is expected to set, and the baseline
# stale-readiness gate (orchestration_service._baseline_readiness_reason)
# reads it back to stop 409-ing Start for a step whose only staleness is a
# suggestion the human has already seen and dismissed.
STALE_INPUTS_DISMISSED_REASON = "dismissed by human"


def _is_valid_human_actor(actor: str) -> bool:
    """Validate human actor attribution format (human:<user_id>)."""
    if not actor or not actor.startswith("human:"):
        return False
    return len(actor) > len("human:")


def _is_valid_resolver_actor(actor: str) -> bool:
    """Validate resolver actor attribution format.

    Accepts:
    - "system" (auto-resolution)
    - "orchestrator" or "orchestrator:<process>" (deterministic execution)
    - "human:<user_id>" (human manual resolution)
    - "manager:<agent_id>" (manager manual resolution)
    """
    if actor == AUTO_RESOLVED_BY:
        return True
    if actor == "orchestrator" or actor.startswith("orchestrator:"):
        return len(actor) == len("orchestrator") or len(actor) > len("orchestrator:")
    if actor.startswith("human:") or actor.startswith("manager:"):
        prefix_len = actor.index(":") + 1
        return len(actor) > prefix_len
    return False


class OrchestrationWarningService:
    """Durable warning lifecycle (Spec 11, 15.3).

    Acknowledgement keeps the warning active — an acknowledged risk stays
    visible (spec 11.9). Resolution deactivates it with who and why.
    Re-evaluation triggers, blocker mirroring, and the completion rule are
    Phase 10; this service only provides the primitives.

    `run_id`, `source_process_run_id`, `related_gate_id`, and
    `related_action_id`, and `related_authority_decision_id` must each belong
    to `goal_id` (Spec Deviation 11):
    the FK alone only proves the referenced row exists, not that it's part
    of this goal's history, so linkage columns are cross-checked before the
    warning is written.
    """

    async def create_warning(
        self,
        db: AsyncSession,
        goal_id: uuid.UUID,
        *,
        warning_type: str,
        severity: str,
        message: str,
        run_id: uuid.UUID | None = None,
        source_process_run_id: uuid.UUID | None = None,
        related_gate_id: uuid.UUID | None = None,
        related_action_id: uuid.UUID | None = None,
        related_agent_id: uuid.UUID | None = None,
        source_agent_review_id: uuid.UUID | None = None,
        related_authority_decision_id: uuid.UUID | None = None,
    ) -> OrchestrationWarning:
        if severity not in WARNING_SEVERITIES:
            raise ValueError(f"unknown severity: {severity!r}")
        if severity == "blocker" and run_id is None:
            raise ValueError("blocker severity warnings require run_id for active_blockers mirroring")
        if db.get_bind().dialect.name == "sqlite":
            # SQLite legacy mode needs DML to acquire its writer lock and
            # start a real transaction. Hold it through the caller's commit.
            await db.execute(
                update(OrchestrationGoal)
                .where(OrchestrationGoal.id == goal_id)
                .values(id=OrchestrationGoal.id, updated_at=OrchestrationGoal.updated_at)
            )
        else:
            # Serialize the entire linkage-check/dedup/insert sequence across
            # sessions without constraining intentional append-only warnings.
            await db.execute(
                select(OrchestrationGoal.id)
                .where(OrchestrationGoal.id == goal_id)
                .with_for_update()
            )
        await check_goal_linkage(
            db,
            goal_id,
            run_id=run_id,
            source_process_run_id=source_process_run_id,
            related_gate_id=related_gate_id,
            related_action_id=related_action_id,
            source_agent_review_id=source_agent_review_id,
            related_authority_decision_id=related_authority_decision_id,
        )
        # Idempotent: a re-triggered evaluation of the same condition must
        # not pile up duplicate active warnings for it. Every linkage column
        # is part of the match — distinct agents/gates/actions/runs behind
        # the same (goal, warning_type, source_process_run_id) are distinct
        # warnings, not retries of one another (Spec Deviation 10).
        #
        # Ruling (Phase 10, Phase 2 carry-over): this dedup match
        # deliberately excludes severity and message. A dedup hit NEVER
        # mutates an existing row's severity/message, even if the caller
        # passed different values -- escalating a condition (e.g. warning
        # -> blocker) requires resolve_warning() on the old row followed by
        # a fresh create_warning() call, producing two distinct auditable
        # rows instead of silently rewriting one (spec 11.9 depends on a
        # warning's severity being stable for its lifetime).
        result = await db.execute(
            select(OrchestrationWarning).where(
                OrchestrationWarning.goal_id == goal_id,
                OrchestrationWarning.warning_type == warning_type,
                OrchestrationWarning.source_process_run_id == source_process_run_id,
                OrchestrationWarning.run_id == run_id,
                OrchestrationWarning.related_gate_id == related_gate_id,
                OrchestrationWarning.related_action_id == related_action_id,
                OrchestrationWarning.related_agent_id == related_agent_id,
                OrchestrationWarning.source_agent_review_id == source_agent_review_id,
                OrchestrationWarning.related_authority_decision_id == related_authority_decision_id,
                OrchestrationWarning.active.is_(True),
            )
        )
        existing = result.scalars().first()
        if existing is not None:
            return existing
        warning = OrchestrationWarning(
            goal_id=goal_id,
            run_id=run_id,
            warning_type=warning_type,
            severity=severity,
            message=message,
            source_process_run_id=source_process_run_id,
            related_gate_id=related_gate_id,
            related_action_id=related_action_id,
            related_agent_id=related_agent_id,
            source_agent_review_id=source_agent_review_id,
            related_authority_decision_id=related_authority_decision_id,
        )
        db.add(warning)
        await db.flush()
        if severity == "blocker":
            # Local import avoids the service import cycle.
            from huddleroom.services.orchestration_service import OrchestrationService

            run = await db.get(OrchestrationRun, run_id)
            if run is not None:
                OrchestrationService._upsert_active_blocker(
                    run,
                    {
                        "kind": f"warning:{warning.id}",
                        "task_id": None,
                        "gate_id": str(related_gate_id) if related_gate_id else None,
                        "reason": message,
                        "warning_id": str(warning.id),
                    },
                )
                await db.flush()
        return warning

    async def acknowledge_warning(
        self,
        db: AsyncSession,
        warning: OrchestrationWarning,
        *,
        acknowledged_by: str,
    ) -> OrchestrationWarning:
        # Spec 11.9: acknowledgement is a human sign-off, never an
        # agent/system attribution — otherwise a warning could be
        # "acknowledged" by the very automation that raised it.
        if not _is_valid_human_actor(acknowledged_by):
            raise ValueError("acknowledged_by must be human attribution (\"human:<user_id>\")")
        # Atomic conditional update: only succeeds if warning is still active.
        # Prevents acknowledging a warning that was concurrently resolved.
        result = await db.execute(
            update(OrchestrationWarning)
            .where(OrchestrationWarning.id == warning.id, OrchestrationWarning.active.is_(True))
            .values(acknowledged_by=acknowledged_by, acknowledged_at=_utcnow())
        )
        if result.rowcount == 0:
            raise ValueError("cannot acknowledge a resolved warning")
        await db.refresh(warning)
        return warning

    async def resolve_warning(
        self,
        db: AsyncSession,
        warning: OrchestrationWarning,
        *,
        resolved_by: str,
        reason: str,
    ) -> OrchestrationWarning:
        # Attribution convention (Phase 1): "system" for auto-resolution, or
        # "human:<user_id>" / "manager:<agent_id>" / "orchestrator[:<process>]"
        # for a manual actor — never blank, never a bare unrecognized string.
        if not _is_valid_resolver_actor(resolved_by):
            raise ValueError(f"invalid resolved_by attribution: {resolved_by!r}")
        if not reason or not reason.strip():
            raise ValueError("reason must not be empty")
        # Atomic conditional update: only succeeds if warning is still active.
        # Prevents concurrent resolutions from overwriting resolver/reason.
        result = await db.execute(
            update(OrchestrationWarning)
            .where(OrchestrationWarning.id == warning.id, OrchestrationWarning.active.is_(True))
            .values(
                active=False,
                resolved_by=resolved_by,
                resolved_reason=reason,
                resolved_at=_utcnow(),
            )
        )
        if result.rowcount == 0:
            raise ValueError("warning is already resolved")
        await db.refresh(warning)
        if warning.severity == "blocker" and warning.run_id is not None:
            from huddleroom.services.orchestration_service import OrchestrationService

            run = await db.get(OrchestrationRun, warning.run_id)
            if run is not None:
                OrchestrationService._remove_active_blocker_by_kind(run, f"warning:{warning.id}")
                await db.flush()
        return warning

    async def suggest_stale_inputs(
        self,
        db: AsyncSession,
        goal_id: uuid.UUID,
        *,
        process_type: str,
        process_run_id: uuid.UUID,
        step_label: str,
        run_id: uuid.UUID | None = None,
    ) -> OrchestrationWarning:
        """One-time "inputs changed, re-run?" suggestion for a process run.

        Looked up WITHOUT an active-only filter: once a suggestion for this
        exact process run has been created (and possibly since dismissed),
        it must never be recreated for that same run -- a dismissed
        suggestion must not resurface on the next tick. A fresh process run
        (from an actual rerun) gets its own row and is eligible again.
        """
        warning_type = f"{process_type}_stale_inputs"
        result = await db.execute(
            select(OrchestrationWarning).where(
                OrchestrationWarning.goal_id == goal_id,
                OrchestrationWarning.warning_type == warning_type,
                OrchestrationWarning.source_process_run_id == process_run_id,
            )
        )
        existing = result.scalars().first()
        if existing is not None:
            return existing
        return await self.create_warning(
            db,
            goal_id,
            warning_type=warning_type,
            severity="recommendation",
            message=f"Definitions changed since {step_label} ran — re-run?",
            run_id=run_id,
            source_process_run_id=process_run_id,
        )

    async def stale_inputs_dismissed(
        self,
        db: AsyncSession,
        goal_id: uuid.UUID,
        *,
        process_type: str,
        process_run_id: uuid.UUID,
    ) -> bool:
        """True iff this process run's stale-inputs suggestion was explicitly
        dismissed by a human (resolved_reason == STALE_INPUTS_DISMISSED_REASON).

        Used by the baseline stale-readiness gate to stop 409-ing Start for
        staleness the human has already seen and consciously dismissed. An
        approved rerun instead supersedes the process run entirely -- a
        fresh run_id means this always re-derives against the new row.
        """
        result = await db.execute(
            select(OrchestrationWarning.id).where(
                OrchestrationWarning.goal_id == goal_id,
                OrchestrationWarning.warning_type == f"{process_type}_stale_inputs",
                OrchestrationWarning.source_process_run_id == process_run_id,
                OrchestrationWarning.active.is_(False),
                OrchestrationWarning.resolved_reason == STALE_INPUTS_DISMISSED_REASON,
            )
        )
        return result.scalars().first() is not None

    async def list_warnings(
        self,
        db: AsyncSession,
        goal_id: uuid.UUID,
        *,
        active_only: bool = False,
    ) -> list[OrchestrationWarning]:
        query = select(OrchestrationWarning).where(OrchestrationWarning.goal_id == goal_id)
        if active_only:
            query = query.where(OrchestrationWarning.active.is_(True))
        query = query.order_by(
            OrchestrationWarning.created_at.asc(),
            OrchestrationWarning.id.asc(),
        )
        result = await db.execute(query)
        return list(result.scalars().all())
